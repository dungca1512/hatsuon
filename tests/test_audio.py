"""Audio preprocessing on synthetic signals (spec §4.0, §2.3)."""

import io

import numpy as np
import pytest

from hatsuon.audio import SR, AudioOptions, load
from hatsuon.errors import AudioError


def tone(sec, amp=0.3, f=220.0, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(sec, amp=1e-4, sr=SR, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(int(sec * sr))).astype(np.float32)


def utterance(sr=SR):
    return np.concatenate([silence(0.5, sr=sr), tone(1.0, sr=sr), silence(0.5, sr=sr, seed=1)])


def test_speech_region_with_margin():
    a = load((utterance(), SR))
    assert a.warnings == []
    s, e = a.speech
    assert 0.5 - 0.2 < s / SR < 0.5 and 1.5 < e / SR < 1.5 + 0.2
    assert a.snr_db > 40


def test_resample_and_stereo():
    x = utterance(sr=44100)
    stereo = np.stack([x, x], axis=1)
    a = load((stereo, 44100))
    assert abs(a.duration - 2.0) < 0.01 and a.wave.dtype == np.float32


def test_decode_wav_bytes():
    sf = pytest.importorskip("soundfile")
    buf = io.BytesIO()
    sf.write(buf, utterance(), SR, format="WAV")
    a = load(buf.getvalue())
    assert a.speech is not None


def test_no_speech():
    assert "no_speech" in load((silence(1.0), SR)).warnings
    assert "no_speech" in load((np.zeros(SR, np.float32), SR)).warnings


def test_quality_warnings():
    assert "too_quiet" in load((utterance() * 0.003, SR)).warnings
    clipped = np.clip(utterance() * 10, -1, 1)
    assert "clipping" in load((clipped, SR)).warnings
    # SNR ≈ 8 dB: still detected as speech, but flagged
    n = silence(2.0, 0.03, seed=2)
    noisy = n + np.concatenate([np.zeros(8000), tone(1.0, 0.1), np.zeros(8000)]).astype(np.float32)
    a = load((noisy, SR))
    assert a.speech is not None and "low_snr" in a.warnings and 5 < a.snr_db < 11


def test_truncated():
    x = np.concatenate([tone(1.0), silence(0.5)])
    assert "truncated" in load((x, SR)).warnings


@pytest.mark.parametrize(
    "bad", [(np.zeros(0, np.float32), SR), (np.full(100, np.nan, np.float32), SR), b"not audio"]
)
def test_hard_errors(bad):
    with pytest.raises(AudioError):
        load(bad)


def test_max_duration():
    with pytest.raises(AudioError, match="limit"):
        load((tone(3.0), SR), AudioOptions(max_duration=2.0))


@pytest.mark.parametrize("sr", [0, 1, 4000, 10_000_000])
def test_implausible_sample_rate_rejected(sr):
    # sr=1 would otherwise be upsampled 16000x before any length check
    with pytest.raises(AudioError, match="sample rate"):
        load((np.zeros(1000, np.float32), sr))


def test_long_tuple_rejected_before_resampling(monkeypatch):
    import hatsuon.audio as audio_mod

    def boom(*a, **k):
        raise AssertionError("resampled an over-long input")

    monkeypatch.setattr(audio_mod, "_to_mono_16k", boom)
    x = np.zeros((3 * 44100, 2), np.float32)
    with pytest.raises(AudioError, match="limit"):
        load((x, 44100), AudioOptions(max_duration=2.0))


def test_long_file_rejected_from_header(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "long.wav"
    sf.write(path, tone(3.0), SR)

    def boom(*a, **k):
        raise AssertionError("decoded samples of an over-long file")

    monkeypatch.setattr(sf, "read", boom)
    with pytest.raises(AudioError, match="limit"):
        load(str(path), AudioOptions(max_duration=2.0))
    with pytest.raises(AudioError, match="limit"):
        load(path.read_bytes(), AudioOptions(max_duration=2.0))


def test_too_many_channels_rejected(tmp_path, monkeypatch):
    with pytest.raises(AudioError, match="channel"):
        load((np.zeros((SR, 16), np.float32), SR))
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "wide.wav"
    sf.write(path, np.zeros((SR // 10, 16), np.float32), SR)
    monkeypatch.setattr(sf, "read", lambda *a, **k: pytest.fail("decoded a 16-channel file"))
    with pytest.raises(AudioError, match="channel"):
        load(str(path))


def test_file_within_limit_still_decodes(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "ok.wav"
    sf.write(path, utterance(sr=44100), 44100)
    a = load(str(path), AudioOptions(max_duration=2.5))
    assert abs(a.duration - 2.0) < 0.01 and a.speech is not None


def test_deterministic():
    a, b = load((utterance(), SR)), load((utterance(), SR))
    assert a.speech == b.speech and np.array_equal(a.wave, b.wave)
