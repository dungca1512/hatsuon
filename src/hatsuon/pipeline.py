"""End-to-end scoring pipeline (spec §3): stages A–H wired together."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from . import audio as audio_mod
from .acoustic import FRAME_SHIFT, get_model
from .align import AlignmentError, edit_align, force_align, greedy_decode
from .audio import SR
from .calibration import Calibration, load_calibration
from .frontend.analyze import Utterance, analyze
from .frontend.morae import MoraKind
from .gop import PhoneGop, phone_gop
from .pitch import PitchOptions, extract, judge_phrase, segment_value
from .targets import PAUSE_TOKENS, build_target
from .timing import TimingBounds, judge, mean_mora_duration
from .types import (
    AccentOutcome,
    Insertion,
    MoraResult,
    MoraStatus,
    PitchStatus,
    ScoreResult,
    Warning,
    WordResult,
)

_STRIDE = int(FRAME_SHIFT * SR)
_QUALITY_FACTOR = {
    Warning.LOW_SNR: 0.7, Warning.CLIPPING: 0.8, Warning.TOO_QUIET: 0.8,
    Warning.TRUNCATED: 0.9, Warning.PARTIAL_READING_UNCERTAIN: 0.9,
}


@dataclass
class _Mora:
    """Working state for one target mora."""

    word: int
    index: int
    slots: list[int]
    score: float = 100.0
    deleted: bool = False
    span: tuple[int, int] | None = None  # frames, relative to the speech region
    worst: PhoneGop | None = None
    pitch_value: float | None = None


def _phone_score(gop: float, mean: float, sd: float, k: float, z0: float) -> float:
    """Logistic in the native z-score, rescaled so a typical native phone (z ≥ 0) = 100."""
    z = min(0.0, (gop - mean) / sd)
    sig = lambda v: 1.0 / (1.0 + math.exp(-k * (v - z0)))
    return 100.0 * sig(z) / sig(0.0)


def run(
    audio,
    text: str,
    *,
    reading=None,
    accent=None,
    reference_audio=None,
    pitch: bool = True,
    timing: bool = True,
    debug: bool = False,
    calibration: Calibration | None = None,
) -> ScoreResult:
    if reference_audio is not None:
        raise NotImplementedError("reference-audio comparison (spec §4.4.6) is not available yet")
    cal = calibration or load_calibration()
    utt = analyze(text, reading, accent)  # text errors first: cheap, no audio decode
    a = audio_mod.load(audio)
    warns = [Warning(w) for w in a.warnings]
    if utt.reading_uncertain:
        warns.append(Warning.PARTIAL_READING_UNCERTAIN)
    if a.speech is None:
        return _no_speech(utt, warns)

    model = get_model()
    tok = model.vocab.tokens
    blank = model.vocab.blank
    s0, s1 = a.speech
    lp = model.logposteriors(a.wave[s0:s1])
    tg = build_target(utt, model.vocab)
    pause_ids = frozenset(model.vocab.ids[p] for p in PAUSE_TOKENS)
    hyp = greedy_decode(lp, blank, drop=pause_ids)
    ed = edit_align([h[0] for h in hyp], tg.slots)
    try:
        al = force_align(lp, tg.slots, blank)
    except AlignmentError:  # far too little audio for the target
        return _no_speech(utt, warns + [Warning.CONTENT_MISMATCH])
    mismatch = ed.match_ratio < cal["content_mismatch_ratio"]
    if mismatch:
        warns.append(Warning.CONTENT_MISMATCH)

    # --- stage E: per-phone GOP → per-mora score / status -----------------------------
    phones = model.vocab.phone_ids()
    gops: dict[int, PhoneGop] = {}
    # every mora gets an entry; one without phonemes (utterance-initial ー) keeps no slots
    morae = {(wi, mi): _Mora(wi, mi, []) for wi, w in enumerate(utt.words)
             for mi in range(len(w.morae))}
    for i, ref in enumerate(tg.refs):
        if ref is None:
            continue
        gops[i] = phone_gop(lp, al.segments[i], tg.slots[i], phones)
        morae[(ref.word, ref.mora)].slots.append(i)

    k, z0 = cal["gop_slope"], cal["gop_z0"]
    for m in morae.values():
        scored = []
        for i in m.slots:
            g = gops[i]
            if g.gop is None:
                continue
            n = cal.gop_norm(tok[tg.slots[i].alts[0]])
            scored.append((_phone_score(g.gop, n.mean, n.sd, k, z0), g))
        segs = [al.segments[i] for i in m.slots if not al.segments[i].skipped]
        if segs:
            m.span = (min(sg.start for sg in segs), max(sg.end for sg in segs))
        required = [i for i in m.slots if not tg.slots[i].optional]
        no_match = all(ed.matched[i] is None for i in required)
        core = [gops[i].core_logp for i in m.slots if gops[i].gop is not None]
        if required and no_match and (not core or np.mean(core) < cal["deletion_core_logp"]):
            m.deleted, m.score = True, 0.0
        elif scored:
            # worst phone: one wrong onset must be able to flag the mora (spec §4.3.2)
            m.score = float(min(sc for sc, _ in scored))
            m.worst = min(scored, key=lambda x: x[0])[1]

    # --- stage E: special-mora timing --------------------------------------------------
    timing_status: dict[tuple[int, int], object] = {}
    timing_ratio: list[dict] = []  # debug / calibration: every measured special mora
    if timing and not mismatch:
        bounds = {
            MoraKind.LONG: TimingBounds(*cal["timing_long"]),
            MoraKind.SOKUON: TimingBounds(*cal["timing_sokuon"]),
            MoraKind.HATSUON: TimingBounds(*cal["timing_hatsuon"]),
        }
        mean = mean_mora_duration([m.span for m in morae.values() if not m.deleted])
        last_key = max(morae) if morae else None
        for key, m in morae.items():
            mo = utt.words[m.word].morae[m.index]
            if mo.kind not in bounds or m.deleted or m.span is None:
                continue
            if mo.kind is MoraKind.SOKUON and key == last_key:
                continue  # word-final っ (exclamation): unscorable, spec §5
            dur = m.span[1] - m.span[0]
            if mo.kind is MoraKind.LONG:
                prev = morae.get((m.word, m.index - 1))
                if prev is None or prev.deleted:
                    continue
                vseg = al.segments[prev.slots[-1]]
                if vseg.skipped:
                    continue
                dur += vseg.end - vseg.start
            reliable = all(gops[i].core_logp > cal["deletion_core_logp"] for i in m.slots
                           if gops[i].gop is not None)
            timing_status[key] = judge(mo.kind, dur, mean, bounds, reliable).status
            if mean:
                timing_ratio.append({"word": key[0], "mora": key[1], "kind": mo.kind.value,
                                     "ratio": dur / mean, "reliable": reliable})

    # insertions: runs of free-decode tokens with no target counterpart
    insertions: list[Insertion] = []
    for idx in ed.insertions:
        q, a_f, b_f = hyp[idx]
        t_a, t_b = (s0 + a_f * _STRIDE) / SR, (s0 + b_f * _STRIDE) / SR
        if insertions and idx - 1 in ed.insertions and t_a - insertions[-1].end < 0.1:
            last = insertions[-1]
            insertions[-1] = Insertion(last.start, t_b, f"{last.content} {tok[q]}")
        else:
            insertions.append(Insertion(round(t_a, 3), round(t_b, 3), tok[q]))

    # --- stages F, G: pitch --------------------------------------------------------------
    popts = PitchOptions(dead_zone=cal["pitch_dead_zone_st"],
                         peak_delay=cal["pitch_peak_delay_morae"],
                         min_coverage=cal["pitch_min_coverage"])
    phrase_outcome: dict[int, AccentOutcome] = {}
    pitch_status: dict[tuple[int, int], PitchStatus] = {}
    pitch_score = None
    track = None
    if pitch and not mismatch:
        track = extract(a.wave, a.speech, popts)
        expected_voiced = 0
        for key, m in morae.items():
            w = utt.words[m.word]
            mo = w.morae[m.index]
            if mo.kind is MoraKind.SOKUON or w.devoiced[m.index]:
                continue
            expected_voiced += 1
            seg = al.segments[m.slots[-1]]  # vowel / N / long portion
            if m.deleted or seg.skipped:
                continue
            m.pitch_value = segment_value(track, s0 + seg.start * _STRIDE,
                                          s0 + seg.end * _STRIDE, popts)
        voiced = sum(m.pitch_value is not None for m in morae.values())
        if track.ref_hz is None or voiced < popts.min_utt_coverage * max(1, expected_voiced):
            warns.append(Warning.PITCH_UNRELIABLE)
        else:
            for pi, ph in enumerate(utt.phrases):
                keys = [(wi, mi) for wi in ph.words for mi in range(len(utt.words[wi].morae))]
                ms = [morae[kk] for kk in keys if kk in morae]
                gate = (not ms or sum(m.deleted for m in ms) * 2 > len(ms)
                        or np.mean([m.score for m in ms]) < cal["mispronounced_threshold"])
                if gate:
                    phrase_outcome[pi] = AccentOutcome.UNKNOWN
                    continue
                vals = [m.pitch_value for m in ms]
                out = judge_phrase(vals, ph.nucleus, ph.assessable, ph.question_final, popts)
                phrase_outcome[pi] = out
                pitch_status.update(_decision_status(keys, vals, ph.nucleus, out))
            judged = [o for o in phrase_outcome.values() if o is not AccentOutcome.UNKNOWN]
            if judged:
                pitch_score = 100.0 * sum(o is AccentOutcome.CORRECT for o in judged) / len(judged)

    # --- stage H: assemble -------------------------------------------------------------
    thr, sub_margin = cal["mispronounced_threshold"], cal["substitution_margin"]
    words: list[WordResult] = []
    for wi, w in enumerate(utt.words):
        ph = utt.phrases[w.phrase]
        mrs: list[MoraResult] = []
        for mi, mo in enumerate(w.morae):
            m = morae[(wi, mi)]
            status = (MoraStatus.DELETED if m.deleted
                      else MoraStatus.MISPRONOUNCED if m.score < thr else MoraStatus.CORRECT)
            sub = None
            if status is MoraStatus.MISPRONOUNCED and m.worst and m.worst.margin > sub_margin:
                sub = tok[m.worst.competitor] if m.worst.competitor is not None else None
            t0 = t1 = None
            if m.span and not m.deleted:
                t0 = round((s0 + m.span[0] * _STRIDE) / SR, 3)
                t1 = round((s0 + m.span[1] * _STRIDE) / SR, 3)
            mrs.append(MoraResult(
                kana=mo.kana, phonemes=list(mo.phonemes), start=t0, end=t1,
                score=round(m.score, 1), status=status, substituted_by=sub,
                duration=timing_status.get((wi, mi)),
                expected_level=ph.pitch[_offset(utt, wi) + mi].level if ph.pitch else None,
                observed_st=None if m.pitch_value is None else round(m.pitch_value, 2),
                pitch_status=pitch_status.get((wi, mi)),
                devoiced_expected=w.devoiced[mi],
            ))
        spans = [(x.start, x.end) for x in mrs if x.start is not None]
        words.append(WordResult(
            surface=w.surface, reading=w.reading,
            score=round(float(np.mean([x.score for x in mrs])), 1) if mrs else 0.0,
            accent_type=ph.nucleus if w.is_head else None,
            accent=phrase_outcome.get(w.phrase, AccentOutcome.UNKNOWN) if w.is_head
            else AccentOutcome.UNKNOWN,
            start=spans[0][0] if spans else None, end=spans[-1][1] if spans else None,
            morae=mrs,
        ))

    all_m = list(morae.values())
    accuracy = float(np.mean([m.score for m in all_m]))
    completeness = 100.0 * sum(not m.deleted for m in all_m) / len(all_m)
    if mismatch:
        completeness = min(completeness, 100.0 * ed.match_ratio)
    timing_score = None
    if timing and not mismatch:
        judged_t = [s for s in timing_status.values() if s is not None]
        base = 100.0 * sum(s.value == "ok" for s in judged_t) / len(judged_t) if judged_t else None
        if base is None and insertions:
            base = 100.0
        if base is not None:
            timing_score = base * max(0.0, 1.0 - cal["insertion_penalty"] * len(insertions))
    weights = cal["weights"]
    subs = {"accuracy": accuracy, "timing": timing_score, "pitch": pitch_score}
    used = {n: v for n, v in subs.items() if v is not None}
    wsum = sum(weights[n] for n in used)
    overall = completeness / 100.0 * sum(weights[n] * v for n, v in used.items()) / wsum

    confidence = ed.match_ratio
    for wn in warns:
        confidence *= _QUALITY_FACTOR.get(wn, 1.0)
    if mismatch:
        confidence = min(confidence, 0.3)

    dbg = None
    if debug:
        dbg = {
            "speech_span_s": [s0 / SR, s1 / SR],
            "free_decode": " ".join(tok[q] for q, _, _ in hyp),
            "match_ratio": ed.match_ratio,
            "phones": [  # raw GOP per target phone, for calibration (tools/calibrate.py)
                {"token": tok[tg.slots[i].alts[0]], "word": r.word, "mora": r.mora,
                 "gop": gops[i].gop, "core_logp": gops[i].core_logp,
                 "deleted": morae[(r.word, r.mora)].deleted}
                for i, r in enumerate(tg.refs) if r is not None
            ],
            "timing_ratios": timing_ratio,
            "alignment": [
                None if sg.skipped else {"token": tok[sg.token],
                                         "start": (s0 + sg.start * _STRIDE) / SR,
                                         "end": (s0 + sg.end * _STRIDE) / SR}
                for sg in al.segments
            ],
            "f0_st": None if track is None else
            [None if np.isnan(v) else round(float(v), 2) for v in track.st],
            "f0_ref_hz": None if track is None else track.ref_hz,
            "provisional_params": cal.provisional(),
        }
    return ScoreResult(
        overall=round(overall, 1), accuracy=round(accuracy, 1),
        completeness=round(completeness, 1),
        pitch=None if pitch_score is None else round(pitch_score, 1),
        timing=None if timing_score is None else round(timing_score, 1),
        confidence=round(confidence, 3), words=words, insertions=insertions,
        warnings=list(dict.fromkeys(warns)), debug=dbg,
    )


def _offset(utt: Utterance, wi: int) -> int:
    """Mora offset of word `wi` inside its accent phrase."""
    ph = utt.phrases[utt.words[wi].phrase]
    return sum(len(utt.words[w].morae) for w in ph.words if w < wi)


def _decision_status(keys, vals, nucleus, outcome) -> dict:
    """Per-mora pitch status at the decision point only (spec §4.4.4)."""
    if outcome is AccentOutcome.UNKNOWN or nucleus is None:
        return {}
    out = {}
    if nucleus == 0:
        if outcome is AccentOutcome.FALL_FOR_UNACCENTED:
            pts = [(i, v) for i, v in enumerate(vals) if v is not None]
            steps = [(j, vj - vi) for (_, vi), (j, vj) in pairwise(pts)]
            j = min(steps, key=lambda s: s[1])[0]
            out[keys[j]] = PitchStatus.TOO_LOW
        return out
    k = nucleus - 1
    if k + 1 < len(keys):
        out[keys[k]] = PitchStatus.OK
        out[keys[k + 1]] = (PitchStatus.OK if outcome is AccentOutcome.CORRECT
                            else PitchStatus.TOO_HIGH)
    return out


def _no_speech(utt: Utterance, warns: list[Warning]) -> ScoreResult:
    """Silence (or far too little audio): every sub-score 0, every mora deleted."""
    words = [
        WordResult(
            surface=w.surface, reading=w.reading, score=0.0,
            accent_type=utt.phrases[w.phrase].nucleus if w.is_head else None,
            accent=AccentOutcome.UNKNOWN, start=None, end=None,
            morae=[MoraResult(kana=m.kana, phonemes=list(m.phonemes), start=None, end=None,
                              score=0.0, status=MoraStatus.DELETED,
                              devoiced_expected=w.devoiced[i])
                   for i, m in enumerate(w.morae)],
        )
        for w in utt.words
    ]
    return ScoreResult(overall=0.0, accuracy=0.0, completeness=0.0, pitch=0.0, timing=0.0,
                       confidence=0.0, words=words, warnings=list(dict.fromkeys(warns)))
