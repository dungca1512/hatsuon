"""End-to-end scoring; skipped unless the exported model and JSUT are present."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("pyopenjtalk")
pytest.importorskip("librosa")

import hatsuon
from hatsuon.acoustic import ModelNotFoundError, get_model
from hatsuon.types import MoraStatus, Warning

try:
    get_model()
except ModelNotFoundError:
    pytest.skip("acoustic model not exported (run hatsuon-export)", allow_module_level=True)

JSUT = Path(__file__).resolve().parents[1] / "data/jsut/jsut_ver1.1/basic5000/wav"
WAV1 = JSUT / "BASIC5000_0001.wav"
TEXT1 = "水をマレーシアから買わなくてはならないのです。"
needs_jsut = pytest.mark.skipif(not WAV1.exists(), reason="JSUT not downloaded")


def test_silence_is_no_speech():
    r = hatsuon.score((np.zeros(16000, dtype=np.float32), 16000), "東京に行きます。")
    assert Warning.NO_SPEECH in r.warnings
    assert r.overall == 0 and r.confidence == 0
    assert all(m.status is MoraStatus.DELETED for w in r.words for m in w.morae)


@needs_jsut
def test_native_read_speech_scores_high():
    r = hatsuon.score(str(WAV1), TEXT1)
    assert r.completeness == 100
    assert r.accuracy > 80
    assert Warning.CONTENT_MISMATCH not in r.warnings


@needs_jsut
def test_deterministic():
    a = hatsuon.score(str(WAV1), TEXT1)
    b = hatsuon.score(str(WAV1), TEXT1)
    assert a == b


@needs_jsut
def test_wrong_text_is_mismatch():
    r = hatsuon.score(str(WAV1), "今日はいい天気ですね。")
    assert Warning.CONTENT_MISMATCH in r.warnings
    assert r.confidence <= 0.3
