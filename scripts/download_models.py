"""
Run once during development to download and convert models to ONNX int8.
Requires: pip install "optimum[exporters]" onnxruntime torch transformers

Output goes to models/ directory — commit via Git LFS.
"""
import subprocess
import hashlib
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export_embedding_model():
    out_dir = MODELS_DIR / "jina-code-onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting jina-embeddings-v2-base-code to ONNX...")
    subprocess.run([
        "optimum-cli", "export", "onnx",
        "--model", "jinaai/jina-embeddings-v2-base-code",
        "--task", "feature-extraction",
        "--optimize", "O2",
        str(out_dir),
    ], check=True)

    print("Quantizing to int8...")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(out_dir), quantization_config=qconfig)

    onnx_file = out_dir / "model.onnx"
    if onnx_file.exists():
        print(f"Embedding model saved to {out_dir}")
        print(f"  model.onnx SHA256: {sha256(onnx_file)}")
    else:
        print(f"Warning: model.onnx not found at expected path. Check {out_dir}")


def export_reranker_model():
    out_dir = MODELS_DIR / "reranker-onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting cross-encoder/ms-marco-MiniLM-L-6-v2 to ONNX...")
    subprocess.run([
        "optimum-cli", "export", "onnx",
        "--model", "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "--task", "text-classification",
        "--optimize", "O2",
        str(out_dir),
    ], check=True)

    print("Quantizing to int8...")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantizer = ORTQuantizer.from_pretrained(str(out_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(out_dir), quantization_config=qconfig)

    onnx_file = out_dir / "model.onnx"
    if onnx_file.exists():
        print(f"Reranker model saved to {out_dir}")
        print(f"  model.onnx SHA256: {sha256(onnx_file)}")
    else:
        print(f"Warning: model.onnx not found. Check {out_dir}")


def write_checksums():
    checksums = {}
    for name in ("jina-code-onnx", "reranker-onnx"):
        model_file = MODELS_DIR / name / "model.onnx"
        if model_file.exists():
            checksums[name] = sha256(model_file)
    out = MODELS_DIR / "checksums.json"
    out.write_text(json.dumps(checksums, indent=2))
    print(f"Checksums written to {out}")


if __name__ == "__main__":
    export_embedding_model()
    export_reranker_model()
    write_checksums()
    print("\nDone. Commit models/ via Git LFS:")
    print("  git lfs track 'models/**/*.onnx'")
    print("  git add models/")
    print("  git commit -m 'chore: add ONNX int8 quantized models via Git LFS'")
