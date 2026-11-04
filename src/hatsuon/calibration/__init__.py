"""Tuned values with their provenance (spec Appendix A.4).

Every number the scorer uses after the front-end lives in `default.json`, each with
its status (`provisional` or `calibrated`), search range, dataset, metric and date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class GopNorm:
    mean: float = 0.0
    sd: float = 1.0


@dataclass(frozen=True)
class Calibration:
    raw: dict

    def __getitem__(self, name: str):
        return self.raw["params"][name]["value"]

    def gop_norm(self, phone: str) -> GopNorm:
        n = self.raw.get("gop_norms", {}).get(phone)
        return GopNorm(n["mean"], n["sd"]) if isinstance(n, dict) else GopNorm()

    def provisional(self) -> list[str]:
        return [k for k, v in self.raw["params"].items() if v.get("status") != "calibrated"]


@lru_cache(maxsize=4)
def load_calibration(path: str | None = None) -> Calibration:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files(__package__).joinpath("default.json").read_text(encoding="utf-8")
    return Calibration(json.loads(text))
