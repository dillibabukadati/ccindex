"""Download BAAI/bge-small-en-v1.5 and export to ONNX."""
import sys, json
from pathlib import Path

OUT_DIR = Path("models/bge-small-onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError as e:
    print(f"Missing deps: {e}\nInstall: pip install transformers torch")
    sys.exit(1)

MODEL_ID = "BAAI/bge-small-en-v1.5"
print(f"Downloading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()
print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

print("Exporting to ONNX...")
dummy = tokenizer(
    ["def foo(): return 42", "navigate to home screen"],
    return_tensors="pt", padding=True, truncation=True, max_length=512,
)

with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy.get("token_type_ids")),
        str(OUT_DIR / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids":         {0: "batch", 1: "seq"},
            "attention_mask":    {0: "batch", 1: "seq"},
            "token_type_ids":    {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
    )

tokenizer.save_pretrained(str(OUT_DIR))

(OUT_DIR / "config.json").write_text(json.dumps({
    "model_id": MODEL_ID, "dim": 384, "max_seq_len": 512, "pooling": "mean",
}, indent=2))

print(f"\nFiles in {OUT_DIR}:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")
