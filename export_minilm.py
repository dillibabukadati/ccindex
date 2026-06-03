"""
Download all-MiniLM-L6-v2 from HuggingFace and export to ONNX.
Produces: models/minilm-l6-onnx/model.onnx + tokenizer.json
"""
import sys
from pathlib import Path

OUT_DIR = Path("models/minilm-l6-onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Step 1: Download all-MiniLM-L6-v2 from HuggingFace...")
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
except ImportError as e:
    print(f"Missing deps: {e}")
    print("Install: pip install transformers torch")
    sys.exit(1)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

print(f"  Downloaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("Step 2: Export to ONNX...")
import torch.onnx

dummy_input = tokenizer(
    ["def hello(): return 'world'", "function greet() { return 'hi'; }"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
)

with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input.get("token_type_ids")),
        str(OUT_DIR / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids":        {0: "batch", 1: "seq"},
            "attention_mask":   {0: "batch", 1: "seq"},
            "token_type_ids":   {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
    )

onnx_size = (OUT_DIR / "model.onnx").stat().st_size / 1e6
print(f"  Saved model.onnx ({onnx_size:.1f} MB)")

print("Step 3: Save tokenizer as tokenizer.json (HuggingFace fast tokenizer)...")
# all-MiniLM-L6-v2 uses a fast tokenizer we can save directly
tokenizer.save_pretrained(str(OUT_DIR))
# We need tokenizer.json specifically; rename if needed
if (OUT_DIR / "tokenizer.json").exists():
    print("  tokenizer.json saved.")
else:
    print("  WARNING: tokenizer.json not found — may need manual export")

# Also save config
import json
config = {
    "model_id": MODEL_ID,
    "dim": 384,
    "max_seq_len": 256,
    "pooling": "mean",
}
(OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

print(f"\nDone! Files in {OUT_DIR}:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")

print("\nNext: python3 bench_minilm.py")
