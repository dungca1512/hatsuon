"""Special-mora duration judgement (spec §4.3.3).

Durations are expressed in units of the speaker's mean mora duration over the
utterance (pauses excluded), so the bounds hold across speaking rates.

- っ: duration of the closure unit `cl`.
- ん: duration of `N`.
- long vowel (ー / doubled vowel): the *whole* vowel, i.e. the previous mora's vowel
  plus the long mora. CTC boundaries inside one continuous vowel are arbitrary, so
  only the sum is meaningful. Expected ≈ 2 mora-vowels; a shortened one ≈ 1.

Bounds are provisional until calibrated on native data (calibration/default.json).
"""

from __future__ import annotations

from dataclasses import dataclass

from .frontend.morae import MoraKind
from .types import DurationStatus


@dataclass(frozen=True)
class TimingBounds:
    lo: float
    hi: float


@dataclass
class MoraTiming:
    ratio: float | None  # duration / mean mora duration; None = abstained
    status: DurationStatus | None


def mean_mora_duration(spans: list[tuple[int, int] | None]) -> float | None:
    """Frames per mora over aligned (non-pause) morae."""
    durs = [b - a for s in spans if s is not None for a, b in [s] if b > a]
    return sum(durs) / len(durs) if durs else None


def judge(
    kind: MoraKind,
    dur: int | None,
    mean: float | None,
    bounds: dict[MoraKind, TimingBounds],
    reliable: bool = True,
) -> MoraTiming:
    """`dur` in frames (for LONG: previous vowel + long mora)."""
    if kind not in bounds or dur is None or not mean or not reliable:
        return MoraTiming(None, None)
    r = dur / mean
    b = bounds[kind]
    status = (
        DurationStatus.TOO_SHORT if r < b.lo
        else DurationStatus.TOO_LONG if r > b.hi
        else DurationStatus.OK
    )
    return MoraTiming(r, status)
