"""
Code-accuracy test: compare jina-code-onnx INT8 vs MiniLM on code-specific queries.
Tests both NL→code AND code→code search patterns.
Uses pre-built index from bench_models.py — rebuilds per model.
"""
import time
import shutil
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import onnxruntime as ort

FITIX = Path("/Users/dillibabukadati/Documents/fitix")
MODEL_ROOT = Path("/Users/dillibabukadati/Documents/ccindex/models")

# Code-specific eval queries (harder than NL queries — technical vocabulary, code patterns)
CODE_QUERIES = [
    # Code snippet queries — can jina understand code structure better?
    ("useEffect useState hook",                         ["hook", "component", "screen", "tsx", "jsx"]),
    ("async await Promise fetch",                       ["api", "fetch", "client", "supabase", "http"]),
    ("interface type Props export",                     ["types", "interface", "props", "tsx"]),
    ("supabase.from select insert",                     ["supabase", "lib", "client", "db"]),
    ("navigation.navigate router.push",                 ["navigation", "router", "tabs", "screen"]),
    ("try catch error throw",                           ["error", "catch", "handler", "util"]),
    ("useState setLoading isLoading",                   ["loading", "state", "screen", "component"]),
    ("export default function component return jsx",    ["component", "screen", "tsx", "jsx"]),
    # Code snippet as query
    ("const { data, error } = await supabase",         ["supabase", "lib", "client"]),
    ("useFocusEffect useCallback",                      ["screen", "tabs", "component", "tsx"]),
]


class QuickBench:
    def __init__(self, name: str, model_file: str):
        self.name = name
        model_dir = MODEL_ROOT / name
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256 if "minilm" in name else 512)
        self.session = ort.InferenceSession(str(model_dir / model_file), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        probe = self._embed(["x"])
        self.dim = probe.shape[1]

    def _embed(self, texts):
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)[0]
        m = mask[:, :, np.newaxis].astype(np.float32)
        pooled = (out * m).sum(axis=1) / m.sum(axis=1).clip(min=1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype(np.float32)


def build_index(bench: QuickBench):
    from ccindex.config import Config
    from ccindex.walker import walk_project
    from ccindex.chunker import chunk_file
    from ccindex.index import Index

    idx_dir = FITIX / ".ccindex"
    if idx_dir.exists():
        shutil.rmtree(idx_dir)

    cfg = Config()
    all_chunks = []
    for p in walk_project(FITIX, cfg):
        all_chunks.extend(chunk_file(p, FITIX, cfg))
    sorted_chunks = sorted(all_chunks, key=lambda c: len(c.chunk_text))

    index = Index(FITIX / ".ccindex" / "index.db", embedding_dim=bench.dim)
    t = time.perf_counter()
    for i in range(0, len(sorted_chunks), 32):
        batch = sorted_chunks[i:i + 32]
        embs = bench._embed([c.chunk_text for c in batch])
        index.upsert_chunks(list(zip(batch, embs)))
    elapsed = time.perf_counter() - t
    print(f"  Indexed {len(all_chunks)} chunks in {elapsed:.0f}s")
    return index


def eval_queries(bench: QuickBench, index):
    h1 = h3 = h5 = 0
    print(f"\n  {'Query':<42} Top-1 result                          Hit")
    print(f"  {'-'*42} {'-'*38} ---")
    for query, keywords in CODE_QUERIES:
        q_vec = bench._embed([query])[0]
        results = index.vector_search(q_vec, top_k=5)
        paths = [r.file_path.lower() for r in results]

        hit1 = any(kw in paths[0] for kw in keywords) if paths else False
        hit3 = any(any(kw in p for kw in keywords) for p in paths[:3])
        hit5 = any(any(kw in p for kw in keywords) for p in paths[:5])
        h1 += hit1; h3 += hit3; h5 += hit5

        top = results[0].file_path if results else "—"
        score = results[0].score if results else 0
        marker = "✓ @1" if hit1 else ("~ @3" if hit3 else ("~ @5" if hit5 else "✗"))
        short_q = query[:41]
        short_p = top[-37:] if len(top) > 37 else top
        print(f"  {short_q:<42} {short_p:<38} {marker}")

    n = len(CODE_QUERIES)
    return {"hits_at_1": h1/n, "hits_at_3": h3/n, "hits_at_5": h5/n}


if __name__ == "__main__":
    configs = [
        ("jina-code-onnx", "model-int8.onnx"),
        ("minilm-l6-onnx",  "model.onnx"),
        ("bge-small-onnx",  "model.onnx"),
    ]
    results = []
    for name, mfile in configs:
        print(f"\n{'='*70}")
        print(f"Model: {name}/{mfile}")
        bench = QuickBench(name, mfile)
        print(f"  dim={bench.dim}, building index...")
        index = build_index(bench)
        r = eval_queries(bench, index)
        results.append((name, r))
        print(f"\n  Hits@1={r['hits_at_1']:.0%}  Hits@3={r['hits_at_3']:.0%}  Hits@5={r['hits_at_5']:.0%}")

    print(f"\n{'='*70}")
    print(f"CODE QUERY ACCURACY SUMMARY")
    print(f"{'Model':<30} {'Hits@1':>7} {'Hits@3':>7} {'Hits@5':>7}")
    print("-"*55)
    for name, r in results:
        print(f"{name:<30} {r['hits_at_1']:>7.0%} {r['hits_at_3']:>7.0%} {r['hits_at_5']:>7.0%}")
    print("="*55)
