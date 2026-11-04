"""Audio preprocessing (spec §4.0): decode, resample, VAD, quality estimates."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from os import PathLike, fspath

import numpy as np

from .errors import AudioError

SR = 16000
_FRAME = 400  # 25 ms analysis window for VAD/quality
_HOP = 160  # 10 ms
_EPS = 1e-10
# input sample-rate bounds: below 8 kHz the model is useless, and an absurd rate
# would make resampling allocate far more memory than the input itself
MIN_SR = 8_000
MAX_SR = 192_000

AudioInput = str | PathLike | bytes | tuple[np.ndarray, int]


@dataclass(frozen=True)
class AudioOptions:
    """Preprocessing knobs. Values marked provisional in calibration/default.json."""

    max_duration: float = 60.0  # seconds; longer input is a hard error
    trim_margin: float = 0.15  # VAD safety margin (spec: 100–200 ms)
    # speech = energy above noise floor + margin; must stay below low_snr_db, or
    # low-SNR speech is classified as no_speech and low_snr can never fire
    vad_margin_db: float = 6.0
    vad_range_db: float = 45.0  # ... and within this range of the loudest frame
    min_speech: float = 0.10  # seconds of speech below which → no_speech
    clip_level: float = 0.999
    clip_ratio: float = 0.001
    quiet_dbfs: float = -40.0
    low_snr_db: float = 10.0
    truncation_guard: float = 0.05  # speech this close to a file edge → truncated


@dataclass
class Audio:
    wave: np.ndarray  # 16 kHz mono float32, DC-removed, original scale
    speech: tuple[int, int] | None  # sample span of the trimmed speech region
    scale: float  # peak-normalization factor (analysis only, never a score feature)
    warnings: list[str] = field(default_factory=list)
    snr_db: float | None = None
    rms_dbfs: float | None = None

    @property
    def duration(self) -> float:
        return len(self.wave) / SR


def _n_samples(x: np.ndarray) -> int:
    # 2-D input is (samples, channels) or (channels, samples); samples is the long axis
    return int(x.shape[0]) if x.ndim == 1 else int(max(x.shape))


def _check_size(n: int, sr: int, max_duration: float) -> None:
    """Reject a bad rate or an over-long input before it is decoded or resampled."""
    if not MIN_SR <= sr <= MAX_SR:
        raise AudioError(f"unsupported sample rate {sr} Hz (allowed {MIN_SR}-{MAX_SR} Hz)")
    if n / sr > max_duration:
        raise AudioError(f"audio is {n / sr:.1f}s, limit is {max_duration:.0f}s")


def _decode(audio: AudioInput, max_duration: float) -> tuple[np.ndarray, int]:
    if isinstance(audio, np.ndarray):
        raise TypeError("pass a raw waveform as (samples, sample_rate)")
    if isinstance(audio, tuple):
        x, sr = audio
        x, sr = np.asarray(x, dtype=np.float32), int(sr)
        if x.ndim not in (1, 2):
            raise AudioError(f"waveform must be 1-D or 2-D, got {x.ndim}-D")
        _check_size(_n_samples(x), sr, max_duration)
        return x, sr
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover
        raise ImportError("decoding audio needs: pip install 'hatsuon[audio]'") from e
    src = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else fspath(audio)
    try:
        info = sf.info(src)  # header only: no samples decoded yet
    except RuntimeError as sf_err:  # libsndfile cannot read e.g. M4A/AAC
        return _decode_librosa(src, max_duration, sf_err)
    _check_size(info.frames, info.samplerate, max_duration)
    if isinstance(src, io.BytesIO):
        src.seek(0)
    try:
        # read at most one frame past the limit, in case the header understates it
        x, sr = sf.read(src, frames=int(max_duration * info.samplerate) + 1,
                        dtype="float32", always_2d=True)
        return x, sr
    except RuntimeError as sf_err:
        return _decode_librosa(src, max_duration, sf_err)


def _decode_librosa(src, max_duration: float, sf_err: Exception) -> tuple[np.ndarray, int]:
    try:
        import librosa

        if isinstance(src, io.BytesIO):
            src.seek(0)
        # stop decoding just past the limit; the caller rejects the overflow
        x, sr = librosa.load(src, sr=None, mono=False, duration=max_duration + 1.0)
        return np.asarray(x, dtype=np.float32).T, int(sr)
    except Exception as e:
        raise AudioError(f"cannot decode audio: {sf_err}") from e


def _to_mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim == 2:
        # (samples, channels) from soundfile; a (channels, samples) tuple input is
        # recognized by having fewer rows than columns
        x = x.mean(axis=1) if x.shape[0] >= x.shape[1] else x.mean(axis=0)
    if sr != SR:
        try:
            import librosa
        except ImportError as e:  # pragma: no cover
            raise ImportError(f"resampling from {sr} Hz needs: pip install 'hatsuon[audio]'") from e
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return np.ascontiguousarray(x, dtype=np.float32)


def frame_energy_db(x: np.ndarray) -> np.ndarray:
    """Per-10 ms frame energy in dBFS."""
    if len(x) < _FRAME:
        x = np.pad(x, (0, _FRAME - len(x)))
    n = 1 + (len(x) - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + _EPS)


def load(audio: AudioInput, opts: AudioOptions = AudioOptions()) -> Audio:
    """Decode and preprocess once; the result is shared by all later stages.

    Size limits are checked before decoding (from the file header) and before
    resampling, so an over-long input fails without a large allocation. Encoded
    ``bytes`` are already in the caller's memory: cap request size upstream.
    """
    x, sr = _decode(audio, opts.max_duration)
    if x.size == 0:
        raise AudioError("audio is empty")
    _check_size(_n_samples(x), sr, opts.max_duration)  # decoded length, not the header's
    if not np.all(np.isfinite(x)):
        raise AudioError("audio contains NaN or inf samples")
    clip = float(np.mean(np.abs(x) >= opts.clip_level))
    x = _to_mono_16k(x, sr)
    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)))
    scale = 1.0 / peak if peak > _EPS else 1.0

    warnings: list[str] = []
    if clip > opts.clip_ratio:
        warnings.append("clipping")

    e = frame_energy_db(x)
    floor = float(np.percentile(e, 10))
    thr = max(floor + opts.vad_margin_db, float(e.max()) - opts.vad_range_db)
    voiced = np.flatnonzero(e > thr)
    if peak <= _EPS or voiced.size * _HOP / SR < opts.min_speech:
        return Audio(wave=x, speech=None, scale=scale, warnings=warnings + ["no_speech"])

    margin = int(opts.trim_margin * SR)
    start = max(0, int(voiced[0]) * _HOP - margin)
    end = min(len(x), int(voiced[-1]) * _HOP + _FRAME + margin)

    guard = opts.truncation_guard * SR
    if voiced[0] * _HOP < guard or len(x) - (voiced[-1] * _HOP + _FRAME) < guard:
        warnings.append("truncated")

    speech_e = e[voiced]
    rms_dbfs = float(10 * np.log10(np.mean(10 ** (speech_e / 10)) + _EPS))
    if rms_dbfs < opts.quiet_dbfs:
        warnings.append("too_quiet")
    noise = e[e <= thr]
    snr = None
    if noise.size >= 10:  # need some non-speech to estimate noise
        snr = rms_dbfs - float(10 * np.log10(np.mean(10 ** (noise / 10)) + _EPS))
        if snr < opts.low_snr_db:
            warnings.append("low_snr")
    return Audio(wave=x, speech=(start, end), scale=scale, warnings=warnings,
                 snr_db=snr, rms_dbfs=rms_dbfs)
