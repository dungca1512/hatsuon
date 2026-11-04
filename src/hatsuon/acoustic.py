"""CTC acoustic model on ONNX Runtime (spec §4.2.1, §6.2).

A model directory holds `model.onnx`, `vocab.json` (token → id) and `meta.json`.
It is produced locally by `hatsuon-export`; weights are never shipped with the
package. Resolution order: explicit path → $HATSUON_MODEL_DIR → default cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .errors import ModelNotFoundError

DEFAULT_REPO = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
DEFAULT_REVISION = "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
FRAME_SHIFT = 0.02  # seconds per output frame (conv stride 320 @ 16 kHz)
_STRIDE = 320
# The model drops tokens at the very start of its input when speech begins without
# leading silence (seen with FP32 torch too, not a quantization artefact). Pad with
# digital silence and crop the frames back. Provisional, see calibration/default.json.
CONTEXT_PAD = 0.3  # seconds, multiple of FRAME_SHIFT
BLANK_TOKEN = "PAD"  # HF CTC heads use the pad token as blank
SPECIAL = frozenset({"PAD", "UNK", "SOS", "EOS"})
PAUSE = frozenset({"pau", "sil"})


def default_model_dir() -> Path:
    env = os.environ.get("HATSUON_MODEL_DIR")
    if env:
        return Path(env)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "hatsuon" / "models" / DEFAULT_REPO.split("/")[-1]


@dataclass(frozen=True)
class Vocab:
    ids: dict[str, int]

    @property
    def blank(self) -> int:
        return self.ids[BLANK_TOKEN]

    @property
    def tokens(self) -> list[str]:
        out = [""] * (max(self.ids.values()) + 1)
        for t, i in self.ids.items():
            out[i] = t
        return out

    def phone_ids(self) -> np.ndarray:
        """Ids of real phone tokens (no blank/special/pause) — the GOP competitor set."""
        return np.array(sorted(i for t, i in self.ids.items() if t not in SPECIAL | PAUSE))


class AcousticModel:
    """Frame-level log-posteriors over the model's token set."""

    def __init__(self, model_dir: str | os.PathLike | None = None, threads: int | None = None):
        d = Path(model_dir) if model_dir else default_model_dir()
        onnx_path = d / "model.onnx"
        if not onnx_path.exists():
            raise ModelNotFoundError(
                f"acoustic model not found in {d}. Create it once with:\n"
                f"  pip install 'hatsuon[export]' && hatsuon-export --out {d}"
            )
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise ImportError("acoustic model needs: pip install 'hatsuon[onnx]'") from e
        self.dir = d
        self.vocab = Vocab(json.loads((d / "vocab.json").read_text(encoding="utf-8")))
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        so = ort.SessionOptions()
        # fixed thread count → reproducible reductions and no oversubscription (§6.2)
        so.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.use_deterministic_compute = True
        self._sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def warmup(self) -> None:
        self.logposteriors(np.zeros(16000, dtype=np.float32))

    def logposteriors(self, wave: np.ndarray, pad: float = CONTEXT_PAD) -> np.ndarray:
        """(T, V) float32 log-softmax posteriors for a 16 kHz mono waveform.

        Frame t covers samples [t*320, t*320+400) of `wave`; T ≈ len(wave) / 320.
        """
        n_pad = round(pad / FRAME_SHIFT)
        p = n_pad * _STRIDE
        x = np.pad(np.asarray(wave, dtype=np.float32), (p, p))
        if len(x) < 400:  # shorter than the receptive field
            x = np.pad(x, (0, 400 - len(x)))
        logits = self._sess.run(None, {self._input: x[None, :]})[0][0]
        logits = logits[n_pad : max(n_pad + 1, len(logits) - n_pad)]
        logits = logits.astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        lp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return lp.astype(np.float32)


@lru_cache(maxsize=4)
def get_model(model_dir: str | None = None, threads: int | None = None) -> AcousticModel:
    """Process-wide cached model (load once, warm up once)."""
    m = AcousticModel(model_dir, threads)
    m.warmup()
    return m
