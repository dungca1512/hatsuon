"""Public entry point. Pipeline stages follow spec §3."""

from __future__ import annotations

from os import PathLike

import numpy as np

from .types import ScoreResult

AudioInput = str | PathLike | bytes | tuple[np.ndarray, int]


def score(
    audio: AudioInput,
    text: str,
    *,
    reading: str | dict[str, str] | None = None,
    accent: dict[str, int] | None = None,
    reference_audio: AudioInput | None = None,
    pitch: bool = True,
    timing: bool = True,
    debug: bool = False,
) -> ScoreResult:
    """Score a learner's reading of `text` (spec §2).

    Args:
        audio: file path, encoded bytes, or a `(waveform, sample_rate)` tuple.
        text: target sentence in normal orthography.
        reading: kana for the whole text, or `{word: kana}` for single words.
        accent: `{word: nucleus}` overrides (0 = heiban, n = fall after mora n).
        reference_audio: native recording (spec §4.4.6) — not available yet.
        pitch, timing: turn the pitch-accent / special-mora stages on or off.
        debug: attach F0, free decode and the alignment path to `result.debug`.

    Raises:
        AudioError, TextError: hard errors (spec §2.3). Soft problems are reported
        in `result.warnings` instead.

    The acoustic model must have been created once with `hatsuon-export`.
    """
    from .pipeline import run

    return run(audio, text, reading=reading, accent=accent, reference_audio=reference_audio,
               pitch=pitch, timing=timing, debug=debug)
