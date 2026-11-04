"""Accent-judgement evaluation on JSUT + jsut-label (spec §7, Appendix A.2 dead-zone row).

For each utterance: score it, take the per-mora semitone values, and match the
front-end's accent phrases to the hand-labelled ones (jsut-label e2e_symbol) by mora
offsets. On phrases where the front-end nucleus equals the label nucleus:

- native accuracy  = P(correct | true nucleus), over non-unknown judgements
- false accept     = P(correct | a wrong nucleus), same audio, every other nucleus
                     for which the fall is assessable; split into adjacent accented
                     nuclei (|Δ| = 1) and the rest (incl. heiban ↔ accented)
- coverage         = share of phrases not judged unknown

Usage: python tools/eval_accent.py --n 300 [--out data/eval/accent.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import hatsuon
from hatsuon.frontend.analyze import analyze
from hatsuon.frontend.pitch_pattern import fall_assessable
from hatsuon.pitch import PitchOptions, judge_phrase
from hatsuon.types import AccentOutcome

ROOT = Path(__file__).resolve().parents[1] / "data"
WAV = ROOT / "jsut/jsut_ver1.1/basic5000/wav"
TRANSCRIPT = ROOT / "jsut/jsut_ver1.1/basic5000/transcript_utf8.txt"
PHONEME = ROOT / "jsut-label/jsut-label-master/e2e_symbol/phoneme.yaml"
VOWELS = set("aiueoAIUEO") | {"N", "cl"}


def read_map(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def label_phrases(seq: str) -> list[tuple[int, int, int]]:
    """(start mora, n morae, nucleus) per labelled accent phrase."""
    phrases, cur_n, nucleus, offset = [], 0, 0, 0
    for sym in seq.split("-"):
        if sym in ("#", "_", "^", "$", "?"):
            if cur_n:
                phrases.append((offset, cur_n, nucleus))
                offset += cur_n
            cur_n, nucleus = 0, 0
        elif sym == "]":
            nucleus = cur_n
        elif sym == "[":
            pass
        elif sym in VOWELS:
            cur_n += 1
    return phrases


def collect(ids: list[str]) -> tuple[list[tuple], dict]:
    """Per-mora values of phrases whose front-end nucleus matches the label."""
    texts = read_map(TRANSCRIPT)
    labels = read_map(PHONEME)
    records = []  # (values, true nucleus, n morae, question_final)
    stats = dict.fromkeys(("utts", "utt_mora_mismatch", "phrases_label", "phrases_matched",
                           "nucleus_agree"), 0)
    for uid in ids:
        text = texts[uid]
        utt = analyze(text)
        lab = label_phrases(labels[uid])
        if sum(n for _, n, _ in lab) != utt.n_morae:
            stats["utt_mora_mismatch"] += 1
            continue
        r = hatsuon.score(str(WAV / f"{uid}.wav"), text, timing=False)
        stats["utts"] += 1
        vals = [m.observed_st for w in r.words for m in w.morae]
        lab_by_start = {s: (n, k) for s, n, k in lab}
        stats["phrases_label"] += len(lab)
        offset = 0
        for ph in utt.phrases:
            n = ph.n_morae
            if offset in lab_by_start and lab_by_start[offset][0] == n:
                stats["phrases_matched"] += 1
                if ph.nucleus == lab_by_start[offset][1]:
                    stats["nucleus_agree"] += 1
                    records.append((vals[offset: offset + n], ph.nucleus, n, ph.question_final))
            offset += n
    return records, stats


def evaluate(records: list[tuple], opts: PitchOptions) -> dict:
    c = dict.fromkeys(("tc", "tn", "hc", "hn", "ac", "an", "f1c", "f1n", "f2c", "f2n"), 0)
    for vals, k, n, q in records:
        o = judge_phrase(vals, k, fall_assessable(n, k), q, opts)
        if o is AccentOutcome.UNKNOWN:
            continue
        ok = o is AccentOutcome.CORRECT
        cls = "h" if k == 0 else "a"
        for key, inc in (("tn", 1), ("tc", ok), (cls + "n", 1), (cls + "c", ok)):
            c[key] += inc
        for k2 in range(n + 1):
            if k2 == k or not fall_assessable(n, k2):
                continue
            o2 = judge_phrase(vals, k2, True, q, opts)
            if o2 is not AccentOutcome.UNKNOWN:
                f = "f1" if abs(k2 - k) == 1 and k and k2 else "f2"
                c[f + "n"] += 1
                c[f + "c"] += o2 is AccentOutcome.CORRECT

    def rate(a: str, b: str) -> float | None:
        return round(c[a] / c[b], 3) if c[b] else None

    return {
        "native_accuracy": rate("tc", "tn"),
        "native_heiban": rate("hc", "hn"),
        "native_accented": rate("ac", "an"),
        "false_accept_adjacent": rate("f1c", "f1n"),
        "false_accept_other": rate("f2c", "f2n"),
        "coverage": round(c["tn"] / len(records), 3) if records else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/accent.json")
    a = ap.parse_args()
    ids = sorted(read_map(TRANSCRIPT))
    random.Random(a.seed).shuffle(ids)
    records, stats = collect(ids[: a.n])
    stats["n_phrases"] = len(records)
    grid = {
        f"delay={pd} dz={dz}": evaluate(records, PitchOptions(dead_zone=dz, peak_delay=pd))
        for pd in (0, 1)
        for dz in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    }
    out = {"stats": stats, "grid": grid}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
