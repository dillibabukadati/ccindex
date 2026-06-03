from __future__ import annotations
from pathlib import Path
import struct
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_PACKAGE_ROOT = Path(__file__).parent.parent.parent
_USER_CACHE = Path.home() / ".ccindex" / "models"


class ModelNotFoundError(Exception):
    pass


def get_model_dir(model_name: str) -> Path:
    for candidate in (_PACKAGE_ROOT / "models" / model_name, _USER_CACHE / model_name):
        if candidate.is_dir() and (candidate / "model.onnx").exists():
            return candidate
    raise ModelNotFoundError(
        f"Model '{model_name}' not found.\n"
        f"Run: ccindex update"
    )


class EmbeddingModel:
    def __init__(self, model_dir: Path):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def embed(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, feed)
        hidden = outputs[0]  # (batch, seq_len, hidden_dim)

        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype(np.float32)


class Reranker:
    def __init__(self, model_dir: Path):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        self.tokenizer.enable_truncation(max_length=512)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        # Pass as sequence pairs so the tokenizer inserts [CLS] query [SEP] passage [SEP]
        encoded = self.tokenizer.encode_batch([[query, p] for p in passages])
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        outputs = self.session.run(None, feed)
        logits = outputs[0]
        scores = logits[:, 1] if logits.shape[1] == 2 else logits[:, 0]
        return (1.0 / (1.0 + np.exp(-scores))).tolist()


def serialize_embedding(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.tolist())
