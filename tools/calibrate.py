"""Fit GOP norms and special-mora timing bounds on native speech (spec Appendix A).

Data: JVS parallel100 (100 speakers). Split by speaker (A.1): speakers whose
number is a multiple of 5 are the test set, the rest tune.

    python tools/calibrate.py collect [--per-speaker 30]   # runs the scorer (slow)
    python tools/calibrate.py synth [--per-speaker 6]     # synthetic substitutions (slow)
    python tools/calibrate.py fit [--write]                # fits, reports, records

`collect` scores every utterance with the library itself (debug=True), so the
features are exactly what the pipeline sees. Utterances whose free decode does not
match the transcript (match ratio < 0.8: JVS has a few misread prompts) are dropped.

GOP scale: native GOP is a point mass at 0 (≈ 98.9 % of tokens; the CTC model is
very peaky), so per-phone z-scores and near-edge percentiles are degenerate. One
scale for every phone instead, chosen so native morae fall below the mispronounced
threshold at FA_TARGET (mora score = its worst phone, as in the pipeline). Detection
is measured on synthetic one-mora substitutions (`synth`, spec B.4): native audio
scored against a reading with one mora swapped for a confusable one (k/g, t/d, r/d,
s/z, n/m, h/b, e/i, o/u, a/o).

Timing bounds: [p1, p99] of native ratios on the tune speakers, clipped to the
search ranges. The long-vowel lower range is wider than the spec's original
1.1–1.6: the ratio's denominator is a whole mora (consonant included), and on the
JSUT HTS labels a whole native long vowel is only 1.06 mean morae (p5 0.66) while a
single vowel — what a shortened long vowel becomes — is 0.44 (p95 1.04).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JVS = ROOT / "data/jvs/jvs_ver1"
FEATS = ROOT / "data/eval/calib_features.jsonl"
CAL = ROOT / "src/hatsuon/calibration/default.json"

FA_TARGET = 0.005  # native morae below the mispronounced threshold
BOOT = 20
MIN_MATCH = 0.8
READ_ERR = 0.3  # share of readers deleting the same word → front-end reading error
SEARCH = {"long": ((0.4, 1.6), (2.5, 4.0)), "sokuon": ((0.3, 1.2), (2.0, 5.0)),
          "hatsuon": ((0.3, 1.0), (1.5, 3.5))}


def is_test(spk: str) -> bool:
    return int(spk[3:]) % 5 == 0


def collect(per_speaker: int, seed: int) -> None:
    import hatsuon

    FEATS.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with FEATS.open("w", encoding="utf-8") as f:
        for spk_dir in sorted(JVS.glob("jvs[0-9][0-9][0-9]")):
            d = spk_dir / "parallel100"
            lines = (d / "transcripts_utf8.txt").read_text(encoding="utf-8").splitlines()
            items = [ln.split(":", 1) for ln in lines if ":" in ln]
            rng.shuffle(items)
            for uid, text in items[:per_speaker]:
                wav = d / "wav24kHz16bit" / f"{uid}.wav"
                if not wav.exists():
                    continue
                try:
                    r = hatsuon.score(str(wav), text, pitch=False, debug=True)
                except Exception as e:  # noqa: BLE001 - keep the sweep going, log it
                    print(spk_dir.name, uid, "error:", e)
                    continue
                dbg = r.debug or {}
                rec = {"spk": spk_dir.name, "uid": uid, "match_ratio": dbg.get("match_ratio"),
                       "warnings": [w.value for w in r.warnings],
                       "phones": dbg.get("phones", []), "timing": dbg.get("timing_ratios", [])}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(spk_dir.name, "done", flush=True)


# --- synthetic substitutions (spec B.4: no L2 recordings needed) ----------------------
# Native audio scored against a reading with ONE mora changed to a confusable one,
# i.e. what the scorer sees when a learner makes that substitution.
_ROWS = {  # consonant → kana for a, i, u, e, o (None: irregular or missing)
    "": "アイウエオ", "k": "カキクケコ", "g": "ガギグゲゴ", "s": "サ_スセソ", "z": "ザ_ズゼゾ",
    "t": "タ__テト", "d": "ダ__デド", "n": "ナニヌネノ", "h": "ハヒ_ヘホ", "b": "バビブベボ",
    "p": "パピプペポ", "m": "マミムメモ", "r": "ラリルレロ",
}
_KANA = {k: (c, v) for c, row in _ROWS.items() for k, v in zip(row, "aiueo") if k != "_"}
_CONS_SWAP = {"k": "g", "g": "k", "t": "d", "d": "r", "r": "d", "s": "z", "z": "s",
              "n": "m", "m": "n", "h": "b", "b": "h", "p": "b"}
_VOWEL_SWAP = {"a": "o", "o": "u", "u": "o", "e": "i", "i": "e"}


def _kana(c: str, v: str) -> str | None:
    k = _ROWS.get(c, "_____")["aiueo".index(v)]
    return None if k == "_" else k


def _substitute(kana: str, rng: random.Random) -> str | None:
    c, v = _KANA[kana]
    options = [_kana(_CONS_SWAP[c], v)] if c in _CONS_SWAP else []
    options.append(_kana(c, _VOWEL_SWAP[v]))
    options = [o for o in options if o]
    return rng.choice(options) if options else None


def synth(per_speaker: int, seed: int) -> None:
    import hatsuon
    from hatsuon.frontend.analyze import analyze

    rng = random.Random(seed)
    recs, _ = _load()
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_spk[r["spk"]].append(r)
    out = FEATS.with_name("calib_synth.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for spk, rs in sorted(by_spk.items()):
            d = JVS / spk / "parallel100"
            texts = dict(ln.split(":", 1) for ln in
                         (d / "transcripts_utf8.txt").read_text(encoding="utf-8").splitlines()
                         if ":" in ln)
            done = 0
            for r in rng.sample(rs, len(rs)):
                if done >= per_speaker:
                    break
                words = analyze(texts[r["uid"]]).words
                surf = [w.surface for w in words]
                cands = [(wi, mi) for wi, w in enumerate(words) if surf.count(w.surface) == 1
                         for mi, m in enumerate(w.morae) if m.kana in _KANA]
                if not cands:
                    continue
                wi, mi = rng.choice(cands)
                new = _substitute(words[wi].morae[mi].kana, rng)
                if new is None:
                    continue
                kana = [m.kana for m in words[wi].morae]
                kana[mi] = new
                try:
                    res = hatsuon.score(str(d / "wav24kHz16bit" / f"{r['uid']}.wav"),
                                        texts[r["uid"]], reading={surf[wi]: "".join(kana)},
                                        pitch=False, timing=False, debug=True)
                except Exception as e:  # noqa: BLE001 - keep the sweep going, log it
                    print(spk, r["uid"], "error:", e)
                    continue
                phones = [p for p in (res.debug or {}).get("phones", [])
                          if (p["word"], p["mora"]) == (wi, mi)]
                f.write(json.dumps({"spk": spk, "uid": r["uid"], "word": wi, "mora": mi,
                                    "from": words[wi].morae[mi].kana, "to": new,
                                    "phones": phones}, ensure_ascii=False) + "\n")
                done += 1
            print(spk, "done", flush=True)


def _load() -> tuple[list[dict], dict]:
    recs = [json.loads(ln) for ln in FEATS.read_text(encoding="utf-8").splitlines()]
    keep = [r for r in recs if (r["match_ratio"] or 0) >= MIN_MATCH
            and "no_speech" not in r["warnings"]]
    bad = _reading_errors(keep)
    for r in keep:
        drop = {w for u, w in bad if u == r["uid"]}
        r["phones"] = [p for p in r["phones"] if p["word"] not in drop]
        r["timing"] = [t for t in r["timing"] if t["word"] not in drop]
    info = {"utterances": len(recs), "kept": len(keep),
            "reading_error_words": sorted(f"{u}#{w}" for u, w in bad),
            "tune_speakers": len({r["spk"] for r in keep if not is_test(r["spk"])}),
            "test_speakers": len({r["spk"] for r in keep if is_test(r["spk"])})}
    return keep, info


def _reading_errors(recs: list[dict]) -> set[tuple[str, int]]:
    """(sentence, word) pairs the front-end evidently misreads.

    JVS speakers all read the same prompts, and the JVS labels were generated from
    text, so they cannot be used to check readings. A word deleted by >= READ_ERR of
    the speakers who read that sentence is a text-level reading error (e.g. 命を受けて
    read イノチ, spoken めい), not a native mispronunciation, and is left out.
    """
    readers: dict[str, set[str]] = defaultdict(set)
    deleted: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in recs:
        readers[r["uid"]].add(r["spk"])
        for p in r["phones"]:
            if p["deleted"]:
                deleted[(r["uid"], p["word"])].add(r["spk"])
    return {k for k, spk in deleted.items()
            if len(readers[k[0]]) >= 3 and len(spk) / len(readers[k[0]]) >= READ_ERR}


def _z_at(score: float, k: float, z0: float) -> float:
    """z at which `_phone_score` (pipeline) equals `score`."""
    sig0 = 1.0 / (1.0 + math.exp(k * z0))
    p = score / 100.0 * sig0
    return z0 - math.log(1.0 / p - 1.0) / k


def _mora_min_gop(phones: list[dict]) -> dict[tuple[int, int], float]:
    """Worst (lowest) GOP per mora; a deleted mora is -inf (always flagged)."""
    out: dict[tuple[int, int], float] = {}
    for p in phones:
        key = (p["word"], p["mora"])
        g = -math.inf if p["deleted"] else p["gop"]
        if g is not None:
            out[key] = min(out.get(key, 0.0), g)
    return out


def fit_gop(recs: list[dict], syn: list[dict], k: float, z0: float, thr: float):
    """One scale for every phone, set so that native morae fall below `thr` at rate
    FA_TARGET on the tune speakers. With a uniform scale and worst-phone mora score,
    "mora < thr" is exactly "worst GOP < scale * z_thr", so the scale is a quantile.
    """
    z_thr = _z_at(thr, k, z0)
    counts: dict[str, int] = defaultdict(int)
    native: dict[bool, list[float]] = {False: [], True: []}
    by_spk: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        mg = list(_mora_min_gop(r["phones"]).values())
        native[is_test(r["spk"])].extend(mg)
        if not is_test(r["spk"]):
            by_spk[r["spk"]].extend(mg)
            for p in r["phones"]:
                counts[p["token"]] += 1

    def scale_for(v) -> float:
        return float(np.quantile(np.array(v), FA_TARGET)) / z_thr

    scale = scale_for(native[False])
    rng = random.Random(0)  # speaker bootstrap: is the scale stable?
    spk = sorted(by_spk)
    boot = [scale_for([g for s_ in rng.choices(spk, k=len(spk)) for g in by_spk[s_]])
            for _ in range(BOOT)]

    def rate(v):
        a = np.array(v)
        return round(float(np.mean(a < scale * z_thr)), 4) if len(a) else None

    det: dict[tuple[bool, str], list[float]] = defaultdict(list)
    for r in syn:
        g = _mora_min_gop(r["phones"]).get((r["word"], r["mora"]))
        if g is None:
            continue
        kind = "consonant" if _KANA[r["from"]][1] == _KANA[r["to"]][1] else "vowel"
        det[(is_test(r["spk"]), kind)].append(g)
        det[(is_test(r["spk"]), "all")].append(g)
    info = {
        "scale": round(scale, 4),
        "scale_boot90": [round(float(x), 3) for x in np.percentile(boot, [5, 95])],
        "z_thr": round(z_thr, 3), "fa_target": FA_TARGET,
        "native_fa": {"tune": rate(native[False]), "test": rate(native[True])},
        "synthetic_detection": {f"{'test' if t else 'tune'}_{kd}": [rate(v), len(v)]
                                for (t, kd), v in sorted(det.items())},
        "tokens": dict(sorted(counts.items())),
    }
    norms = {ph: {"mean": 0.0, "sd": round(scale, 4), "n": n} for ph, n in sorted(counts.items())}
    return norms, info


def native_mora_scores(recs: list[dict], norms: dict, k: float, z0: float, test: bool):
    """Per-speaker mora scores (worst phone per mora), as the pipeline scores."""

    def score(g: float, sd: float) -> float:
        z = min(0.0, g / sd)

        def sig(v: float) -> float:
            return 1.0 / (1.0 + math.exp(-k * (v - z0)))

        return 100.0 * sig(z) / sig(0.0)

    per_spk: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        if is_test(r["spk"]) != test:
            continue
        morae: dict[tuple[int, int], list[float]] = defaultdict(list)
        for p in r["phones"]:
            key = (p["word"], p["mora"])
            if p["deleted"]:
                morae[key] = [0.0]
            elif p["gop"] is not None:
                sd = norms.get(p["token"], {}).get("sd", 1.0)
                morae[key].append(score(p["gop"], sd))
        per_spk[r["spk"]].extend(float(min(v)) for v in morae.values() if v)  # = pipeline
    return per_spk


def fit_timing(recs: list[dict]) -> dict:
    out = {}
    for kind, ((lo_a, lo_b), (hi_a, hi_b)) in SEARCH.items():
        tune = [t["ratio"] for r in recs if not is_test(r["spk"]) for t in r["timing"]
                if t["kind"] == kind and t["reliable"]]
        test = np.array([t["ratio"] for r in recs if is_test(r["spk"]) for t in r["timing"]
                         if t["kind"] == kind and t["reliable"]])
        if len(tune) < 50:
            out[kind] = {"n_tune": len(tune), "skipped": "too few tokens"}
            continue
        p1, p99 = np.percentile(tune, [1, 99])
        lo = float(np.clip(round(p1, 2), lo_a, lo_b))
        hi = float(np.clip(round(p99, 2), hi_a, hi_b))
        out[kind] = {
            "value": [lo, hi], "n_tune": len(tune), "n_test": len(test),
            "tune_p1_p50_p99": [round(float(x), 3) for x in np.percentile(tune, [1, 50, 99])],
            "test_false_alarm": round(float(np.mean((test < lo) | (test > hi))), 4)
            if len(test) else None,
        }
    return out


def fit(write: bool) -> None:
    recs, info = _load()
    cal = json.loads(CAL.read_text(encoding="utf-8"))["params"]
    k, z0 = cal["gop_slope"]["value"], cal["gop_z0"]["value"]
    thr = cal["mispronounced_threshold"]["value"]
    syn_path = FEATS.with_name("calib_synth.jsonl")
    syn = ([json.loads(ln) for ln in syn_path.read_text(encoding="utf-8").splitlines()]
           if syn_path.exists() else [])
    bad = set(info["reading_error_words"])
    syn = [r for r in syn if f"{r['uid']}#{r['word']}" not in bad]
    norms, gop_info = fit_gop(recs, syn, k, z0, thr)
    report = {"data": info, "gop": gop_info}
    for name, test in (("tune", False), ("test", True)):
        per = native_mora_scores(recs, norms, k, z0, test)
        means = [float(np.mean(v)) for v in per.values()]
        allv = np.array([x for v in per.values() for x in v])
        report[f"native_{name}"] = {
            "speakers": len(per), "mean_mora_score": round(float(allv.mean()), 2),
            "speaker_mean_min": round(min(means), 2), "speaker_mean_sd": round(float(np.std(means)), 2),
            "mora_false_alarm": round(float(np.mean(allv < thr)), 4),
            "mora_lt70": round(float(np.mean(allv < 70)), 4),
        }
    report["timing"] = fit_timing(recs)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if write:
        _write(norms, report)


def _write(norms: dict, report: dict) -> None:
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    today = date.today().isoformat()
    ds = (f"JVS parallel100, {report['data']['kept']} utts, "
          f"{report['data']['tune_speakers']} tune / {report['data']['test_speakers']} test speakers")
    t, g = report["native_test"], report["gop"]
    cal["gop_norms"] = {
        "_status": "calibrated",
        "_method": f"one scale for all phones (center 0): the {FA_TARGET:.1%} quantile of the "
                   f"worst native GOP per mora (tune speakers) / z at the mispronounced "
                   f"threshold, so native morae fall below it at that rate. Native GOP is a "
                   f"point mass at 0, so per-phone z-scores are degenerate",
        "_dataset": ds + "; synthetic one-mora substitutions on the same audio (spec B.4)",
        "_date": today,
        "_metric": f"scale {g['scale']} (speaker bootstrap 90% {g['scale_boot90']}); test "
                   f"speakers: native mora < threshold {g['native_fa']['test']:.2%}, synthetic "
                   f"substitution detected {g['synthetic_detection'].get('test_all', ['?'])[0]}; "
                   f"native mean mora score {t['mean_mora_score']}",
        **{ph: {"mean": n["mean"], "sd": n["sd"], "n": n["n"]} for ph, n in norms.items()},
    }
    for kind, res in report["timing"].items():
        if "value" not in res:
            continue
        p = cal["params"][f"timing_{kind}"]
        p.update({
            "value": res["value"], "status": "calibrated", "dataset": ds, "date": today,
            "range": [list(r) for r in SEARCH[kind]],
            "metric": f"tune p1/p50/p99 {res['tune_p1_p50_p99']} (n={res['n_tune']}); "
                      f"test native false-alarm {res['test_false_alarm']:.2%} (n={res['n_test']})",
            "note": p.get("note", "") + "; native-only: L2 F1 not yet measured",
        })
    CAL.write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", CAL)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--per-speaker", type=int, default=30)
    c.add_argument("--seed", type=int, default=0)
    sy = sub.add_parser("synth")
    sy.add_argument("--per-speaker", type=int, default=6)
    sy.add_argument("--seed", type=int, default=1)
    f = sub.add_parser("fit")
    f.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.per_speaker, a.seed)
    elif a.cmd == "synth":
        synth(a.per_speaker, a.seed)
    else:
        fit(a.write)


if __name__ == "__main__":
    main()
