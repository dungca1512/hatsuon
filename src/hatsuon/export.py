"""`hatsuon-export`: convert the public HF CTC model to (INT8) ONNX, locally.

Needs the `export` extra (torch, transformers, onnx). Run once per machine; the
runtime only needs onnxruntime afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from .acoustic import DEFAULT_REPO, DEFAULT_REVISION, FRAME_SHIFT, default_model_dir


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(repo: str, revision: str, out: Path, quantize: bool = True, opset: int = 17) -> Path:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForCTC, AutoTokenizer

    model = AutoModelForCTC.from_pretrained(repo, revision=revision).eval()
    vocab = AutoTokenizer.from_pretrained(repo, revision=revision).get_vocab()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).logits

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        torch.onnx.export(
            LogitsOnly(model), (torch.zeros(1, 16000),), str(fp32),
            input_names=["input_values"], output_names=["logits"],
            dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
            opset_version=opset, dynamo=False,
        )
        target = out / "model.onnx"
        if quantize:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            # MatMul/Gemm only: ConvInteger is slow on CPU and hurts the conv front-end
            quantize_dynamic(str(fp32), str(target), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gemm"])
        else:
            fp32.replace(target)

    # sanity check: ONNX vs torch frame-level argmax agreement on noise
    x = np.random.default_rng(0).standard_normal((1, 48000)).astype(np.float32) * 0.05
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).logits[0].numpy()
    got = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(
        None, {"input_values": x})[0][0]
    agree = float(np.mean(ref.argmax(1) == got.argmax(1)))

    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), "utf-8")
    meta = {
        "repo": repo, "revision": revision, "quantized": quantize, "opset": opset,
        "frame_shift": FRAME_SHIFT, "sha256": _sha256(target),
        "argmax_agreement_vs_torch": round(agree, 4),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), "utf-8")
    return target


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hatsuon-export", description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--out", type=Path, default=default_model_dir())
    p.add_argument("--no-quantize", action="store_true", help="keep FP32 weights")
    a = p.parse_args(argv)
    path = export(a.repo, a.revision, a.out, quantize=not a.no_quantize)
    meta = json.loads((a.out / "meta.json").read_text())
    print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB), "
          f"argmax agreement vs torch: {meta['argmax_agreement_vs_torch']:.3f}")


if __name__ == "__main__":
    main()
