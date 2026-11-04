"""Tune `truncation_edge_frac` (spec §2.3 `truncated`, Appendix A) on synthetic cuts.

Only files with natural silence (>= --min-gap) at both ends are used; each one
yields four cuts, one edge at a time:
- negative, "tight": cut 0-30 ms after the VAD speech end (or before its start),
  i.e. the recording stops right after speech has decayed — not truncated;
- positive, "mid-speech": cut inside a speech frame 60-500 ms before the speech
  end (or after its start) — truncated.
For every cut it records the outermost-frame level (`audio.edge_levels`: 0 at the
VAD threshold, 1 at the median speech frame); a cut is flagged at X when that
level >= X. Speakers are split 80/20 (tune/test) by a hash of the speaker id.

Usage:
    python tools/tune_truncation.py cuts --jvs data/jvs/jvs_ver1 \
        --l2 data/l2_practice/manifest.tsv --out data/eval/truncation_cuts.jsonl
    python tools/tune_truncation.py sweep data/eval/truncation_cuts.jsonl
    python tools/tune_truncation.py apply data/l2_practice/manifest.tsv --edge-frac 0.5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from hatsuon import audio
from hatsuon.audio import _FRAME, _HOP, SR, AudioOptions
from hatsuon.errors import AudioError

OPTS = AudioOptions(trim_margin=0.0)


def _split(speaker: str) -> str:
    return "test" if int(hashlib.md5(speaker.encode()).hexdigest(), 16) % 5 == 0 else "tune"


def _level(x: np.ndarray, edge: int) -> float | None:
    e = audio.frame_energy_db(x)
    thr, voiced = audio.vad(e, OPTS)
    if voiced.size * _HOP / SR < OPTS.min_speech:
        return None
    return audio.edge_levels(e, thr, voiced, len(x), OPTS)[edge]


def _cuts(x: np.ndarray, rng: np.random.Generator, min_gap: float):
    e = audio.frame_energy_db(x)
    _, voiced = audio.vad(e, OPTS)
    if voiced.size * _HOP / SR < OPTS.min_speech:
        return
    s0, s1 = int(voiced[0]) * _HOP, int(voiced[-1]) * _HOP + _FRAME
    if s0 < min_gap * SR or len(x) - s1 < min_gap * SR:
        return
    def tight() -> int:
        return int(rng.uniform(0, 0.03) * SR)

    yield "trail", "neg", x[: s1 + tight()]
    yield "lead", "neg", x[max(0, s0 - tight()):]
    vset = set(voiced.tolist())
    lo, hi = int(0.06 * SR / _HOP), int(0.5 * SR / _HOP)
    tail = [i for i in range(int(voiced[-1]) - hi, int(voiced[-1]) - lo) if i in vset]
    head = [i for i in range(int(voiced[0]) + lo, int(voiced[0]) + hi) if i in vset]
    if tail:
        i = int(rng.choice(tail))
        yield "trail", "pos", x[: i * _HOP + int(rng.integers(_HOP, _FRAME))]
    if head:
        i = int(rng.choice(head))
        yield "lead", "pos", x[i * _HOP + int(rng.integers(0, _FRAME - _HOP)):]


def _jvs_files(root: Path, per_speaker: int, rng: np.random.Generator):
    for spk in sorted(p for p in root.glob("jvs[0-9]*") if p.is_dir()):
        wavs = sorted((spk / "parallel100" / "wav24kHz16bit").glob("*.wav"))
        for w in rng.permutation(len(wavs))[:per_speaker]:
            yield "jvs", spk.name, wavs[int(w)]


def _l2_files(manifest: Path, limit: int, rng: np.random.Generator):
    rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
    for k in rng.permutation(len(rows))[:limit]:
        r = rows[int(k)]
        yield "l2", r["speaker"], manifest.parent / r["file"]


def cmd_cuts(a) -> None:
    rng = np.random.default_rng(0)
    sources = []
    if a.jvs:
        sources += list(_jvs_files(a.jvs, a.jvs_per_speaker, rng))
    if a.l2:
        sources += list(_l2_files(a.l2, a.l2_limit, rng))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    used = {"jvs": 0, "l2": 0}
    with a.out.open("w", encoding="utf-8") as f:
        for n, (src, spk, path) in enumerate(sources, 1):
            try:
                x = audio.load(path, OPTS).wave
            except AudioError:  # undecodable files are skipped
                continue
            seed = int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16)
            got = False
            for edge, kind, cut in _cuts(x, np.random.default_rng(seed), a.min_gap):
                lv = _level(cut, 0 if edge == "lead" else 1)
                f.write(json.dumps({"src": src, "split": _split(spk), "edge": edge,
                                    "kind": kind, "level": lv}) + "\n")
                got = True
            used[src] += got
            if n % 500 == 0:
                print(f"{n}/{len(sources)} files, used {used}", flush=True)
    print(f"done: used {used}")


def _rates(recs: list[dict], x: float | None) -> tuple[float, float, int, int]:
    """(false-alarm rate on neg, recall on pos, n_neg, n_pos); x=None → guard only."""
    def flag(r):
        return r["level"] is not None and (x is None or r["level"] >= x)
    neg = [flag(r) for r in recs if r["kind"] == "neg"]
    pos = [flag(r) for r in recs if r["kind"] == "pos"]
    return float(np.mean(neg)), float(np.mean(pos)), len(neg), len(pos)


def cmd_sweep(a) -> None:
    recs = [json.loads(line) for line in a.cuts.open(encoding="utf-8")]
    groups = {}
    for split in ("tune", "test"):
        for src in ("jvs", "l2", "all"):
            g = [r for r in recs if r["split"] == split and (src == "all" or r["src"] == src)]
            if g:
                groups[f"{split}/{src}"] = g
    print("x     " + "  ".join(f"{k:>16}" for k in groups) + "   (FA / recall)")
    fa, rc, *_ = zip(*(_rates(g, None) for g in groups.values()))
    print("guard " + "  ".join(f"{f:7.3f} /{r:6.3f}" for f, r in zip(fa, rc)))
    for x in np.round(np.arange(0, 1.01, 0.1), 2):
        fa, rc, *_ = zip(*(_rates(g, float(x)) for g in groups.values()))
        print(f"{x:4.1f}  " + "  ".join(f"{f:7.3f} /{r:6.3f}" for f, r in zip(fa, rc)))
    for k, g in groups.items():
        _, _, nn, npos = _rates(g, 0)
        print(f"{k}: n_neg={nn} n_pos={npos}")


def cmd_apply(a) -> None:
    rows = list(csv.DictReader(a.manifest.open(encoding="utf-8"), delimiter="\t"))
    old = new = n = 0
    opts = AudioOptions(truncation_edge_frac=a.edge_frac)
    guard_only = AudioOptions(truncation_edge_frac=-1e9)
    for r in rows:
        try:
            x = audio.load(a.manifest.parent / r["file"]).wave
        except AudioError:
            continue
        n += 1
        old += "truncated" in audio.load((x, SR), guard_only).warnings
        new += "truncated" in audio.load((x, SR), opts).warnings
    print(json.dumps({"n": n, "edge_frac": a.edge_frac, "guard_only_rate": round(old / n, 4),
                      "new_rate": round(new / n, 4)}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cuts")
    c.add_argument("--jvs", type=Path)
    c.add_argument("--jvs-per-speaker", type=int, default=20)
    c.add_argument("--l2", type=Path)
    c.add_argument("--l2-limit", type=int, default=4000)
    c.add_argument("--min-gap", type=float, default=0.15)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sweep")
    s.add_argument("cuts", type=Path)
    ap = sub.add_parser("apply")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--edge-frac", type=float, required=True)
    a = p.parse_args()
    {"cuts": cmd_cuts, "sweep": cmd_sweep, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
