"""F0 extraction and accent-phrase judgement (spec §4.4).

F0 is computed once per request on the speech region with librosa's pYIN:
pass 1 (wide range, coarse hop) only estimates the speaker median; pass 2 (fine
hop) searches a window around it, which suppresses octave errors and is cheaper.
All comparisons are in semitones relative to the speaker median.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .audio import SR
from .types import AccentOutcome

HOP = 160  # 10 ms


@dataclass(frozen=True)
class PitchOptions:
    fmin: float = 60.0  # pass-1 search range (Hz)
    fmax: float = 500.0
    window_oct: float = 1.0  # pass-2 range: median × 2^(±window_oct)
    jump_st: float = 6.0  # octave-jump filter: deviation from the local median
    min_voiced: int = 3  # frames per mora below which the mora gets no value
    onset_skip: float = 0.25  # fraction of a vowel span skipped (micro-prosody)
    dead_zone: float = 1.5  # semitones: smaller steps count as level
    peak_delay: int = 1  # morae the accent fall may lag behind the nucleus
    min_coverage: float = 0.5  # phrase: fraction of morae needing a value
    min_utt_coverage: float = 0.4  # utterance: below this → pitch_unreliable


@dataclass
class F0Track:
    st: np.ndarray  # semitones vs. speaker median, NaN = unvoiced; 10 ms frames
    offset: int  # sample index of frame 0 in the full waveform
    ref_hz: float | None  # speaker median


def _pyin(x, fmin, fmax, hop, frame_length=1024):
    import librosa

    # pYIN's HMM voicing decision; its per-frame voiced_prob is not calibrated
    # (median ≈ 0.02 on clean voiced speech), so thresholding it drops most frames.
    f0, voiced, _ = librosa.pyin(
        x, fmin=fmin, fmax=fmax, sr=SR, frame_length=frame_length, hop_length=hop, center=True
    )
    return np.where(voiced, f0, np.nan)


def extract(wave: np.ndarray, span: tuple[int, int], opts: PitchOptions = PitchOptions()) -> F0Track:
    s, e = span
    x = wave[s:e].astype(np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    n = 1 + len(x) // HOP
    empty = F0Track(np.full(n, np.nan), s, None)
    if peak <= 0 or len(x) < 2048:
        return empty
    x = x / peak
    f0 = _pyin(x, opts.fmin, opts.fmax, 2 * HOP)
    if np.all(np.isnan(f0)):
        return empty
    med = float(np.nanmedian(f0))
    lo = max(opts.fmin * 0.8, med * 2 ** -opts.window_oct)
    hi = min(opts.fmax * 1.4, med * 2 ** opts.window_oct)
    f0 = _pyin(x, lo, hi, HOP)
    if np.all(np.isnan(f0)):
        return empty
    ref = float(np.nanmedian(f0))
    st = 12 * np.log2(f0 / ref)
    # octave-jump filter: drop frames far from their local (±2 frame) median
    pad = np.pad(st, 2, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 5)
    with warnings.catch_warnings():  # all-NaN windows are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        local = np.nanmedian(win, axis=1)
    st = np.where(np.abs(st - local) > opts.jump_st, np.nan, st)
    return F0Track(st[:n], s, ref)


def segment_value(track: F0Track, start_s: int, end_s: int, opts: PitchOptions) -> float | None:
    """Median semitone value over the later part of a sample span (vowel portion)."""
    a = (start_s - track.offset) / HOP
    b = (end_s - track.offset) / HOP
    a = a + (b - a) * opts.onset_skip
    lo, hi = max(0, int(np.ceil(a))), min(len(track.st), int(np.floor(b)) + 1)
    if hi <= lo:
        return None
    v = track.st[lo:hi]
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) >= opts.min_voiced else None


def judge_phrase(
    values: list[float | None],
    nucleus: int | None,
    assessable: bool,
    question_final: bool,
    opts: PitchOptions = PitchOptions(),
) -> AccentOutcome:
    """Outcome of one accent phrase from its per-mora semitone values.

    `nucleus` is 1-based (0 = heiban). The fall is expected between mora `nucleus`
    and `nucleus + 1`, but native F0 peaks are often delayed into the next mora
    (JSUT: about half of atamadaka phrases), so steps starting up to `peak_delay`
    morae later also count. A null mora (e.g. devoiced nucleus) is bridged: the
    step to the next voiced mora counts, which accepts the shifted fall.
    """
    if nucleus is None or not assessable:
        return AccentOutcome.UNKNOWN
    vals = list(values)
    if question_final and vals:
        vals[-1] = None  # boundary rise masks the accent fall
    pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(pts) < 2 or len(pts) < opts.min_coverage * len(vals):
        return AccentOutcome.UNKNOWN
    steps = [(i, j, vj - vi) for (i, vi), (j, vj) in pairwise(pts)]
    falls = [d for _, _, d in steps if d < -opts.dead_zone]
    if nucleus == 0:
        return AccentOutcome.FALL_FOR_UNACCENTED if falls else AccentOutcome.CORRECT
    k = nucleus - 1  # 0-based index of the nucleus mora; fall is k → k+1
    window = [d for i, j, d in steps if j > k and i <= k + opts.peak_delay]
    if not window:
        return AccentOutcome.UNKNOWN
    if min(window) < -opts.dead_zone:
        return AccentOutcome.CORRECT
    return AccentOutcome.WRONG_NUCLEUS if falls else AccentOutcome.FLAT_FOR_ACCENTED
