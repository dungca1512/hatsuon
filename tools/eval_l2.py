"""Run `hatsuon.score()` over a manifest of L2 recordings and summarize (spec §7).

The manifest is a TSV with columns `file`, `speaker`, `sentence` and optionally
`ref_score` (an external score, used for comparison only). Results are appended
to a JSONL file, so an interrupted run resumes where it stopped.

Usage:
    python tools/eval_l2.py run data/l2_practice/manifest.tsv data/eval/l2.jsonl [--workers 2] [--limit N]
    python tools/eval_l2.py summary data/eval/l2.jsonl [--out data/eval/l2_summary.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SUBSCORES = ("overall", "accuracy", "completeness", "pitch", "timing", "confidence")


def _score_one(job: tuple[str, str, str]) -> dict:
    import hatsuon

    file, path, sentence = job
    out: dict = {"file": file}
    try:
        r = hatsuon.score(path, sentence)
    except Exception as e:  # noqa: BLE001 — hard errors and crashes are part of the result
        out["error"] = type(e).__name__
        out["message"] = str(e)[:200]
        return out
    for k in SUBSCORES:
        out[k] = getattr(r, k)
    out["warnings"] = [w.value for w in r.warnings]
    out["n_morae"] = sum(len(w.morae) for w in r.words)
    out["n_deleted"] = sum(m.status.value == "deleted" for w in r.words for m in w.morae)
    out["n_mispronounced"] = sum(
        m.status.value == "mispronounced" for w in r.words for m in w.morae
    )
    return out


def run(manifest: Path, out: Path, workers: int, limit: int | None) -> None:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8"), delimiter="\t"))
    done = set()
    if out.exists():
        done = {json.loads(line)["file"] for line in out.open(encoding="utf-8")}
    base = manifest.parent
    jobs = [(r["file"], str(base / r["file"]), r["sentence"]) for r in rows
            if r["file"] not in done][:limit]
    print(f"{len(done)} already scored, {len(jobs)} to go", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # each process runs its own ONNX session with up to 4 intra-op threads (acoustic.py)
    with out.open("a", encoding="utf-8") as f, ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(_score_one, jobs, chunksize=8), 1):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                f.flush()
                print(f"{i}/{len(jobs)}", flush=True)


def _quantiles(x: list[float]) -> dict | None:
    if not x:
        return None
    a = np.asarray(x, float)
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {"n": len(a), "mean": round(float(a.mean()), 1),
            **{f"p{p}": round(float(v), 1) for p, v in zip((10, 25, 50, 75, 90), q)}}


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def summary(results: Path, manifest: Path | None) -> dict:
    res = [json.loads(line) for line in results.open(encoding="utf-8")]
    meta = {}
    if manifest and manifest.exists():
        meta = {r["file"]: r for r in csv.DictReader(manifest.open(encoding="utf-8"),
                                                     delimiter="\t")}
    ok = [r for r in res if "error" not in r]
    errors = Counter(r["error"] for r in res if "error" in r)
    warns = Counter(w for r in ok for w in r["warnings"])
    clean = [r for r in ok if not {"no_speech", "content_mismatch"} & set(r["warnings"])]

    out: dict = {
        "n": len(res),
        "hard_errors": dict(errors),
        "warning_rate": {w: round(c / max(len(ok), 1), 4) for w, c in warns.most_common()},
        "subscores_all": {k: _quantiles([r[k] for r in ok if r[k] is not None])
                          for k in SUBSCORES},
        "subscores_clean": {k: _quantiles([r[k] for r in clean if r[k] is not None])
                            for k in SUBSCORES},
        "mora_rates_clean": {
            "deleted": round(sum(r["n_deleted"] for r in clean)
                             / max(sum(r["n_morae"] for r in clean), 1), 4),
            "mispronounced": round(sum(r["n_mispronounced"] for r in clean)
                                   / max(sum(r["n_morae"] for r in clean), 1), 4),
        },
    }
    if meta:
        spk = Counter(meta[r["file"]]["speaker"] for r in clean if r["file"] in meta)
        out["speakers_clean"] = len(spk)
        pairs = [(r["overall"], float(meta[r["file"]]["ref_score"])) for r in clean
                 if meta.get(r["file"], {}).get("ref_score") not in (None, "")]
        if pairs:
            xs, ys = zip(*pairs)
            out["ref_score_comparison"] = {
                "n": len(pairs),
                "spearman_overall": _spearman(list(xs), list(ys)),
                "ref_score": _quantiles(list(ys)),
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("manifest", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--limit", type=int, help="score only the first N remaining files")
    s = sub.add_parser("summary")
    s.add_argument("results", type=Path)
    s.add_argument("--manifest", type=Path)
    s.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.manifest, a.out, a.workers, a.limit)
    else:
        result = summary(a.results, a.manifest)
        text = json.dumps(result, ensure_ascii=False, indent=1)
        if a.out:
            a.out.write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
