"""
Full benchmark: jina-code-onnx (INT8) vs all-MiniLM-L6-v2 on the fitix project.
Tests: index speed, model size, query latency, search quality.

Run: python3 bench_models.py
Results saved to: bench_results.md
"""
import time
import shutil
import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import onnxruntime as ort

FITIX = Path("/Users/dillibabukadati/Documents/fitix")
MODEL_ROOT = Path("/Users/dillibabukadati/Documents/ccindex/models")
RESULTS_FILE = Path("bench_results.md")

# Evaluation queries with expected keywords in the top result's file path
EVAL_QUERIES = [
    # (query, expected_path_keywords)
    ("user authentication login",          ["auth", "login", "oauth", "session"]),
    ("workout exercise tracking",          ["workout", "exercise", "training", "fitness"]),
    ("navigation tabs screens",            ["tabs", "navigation", "screen", "layout"]),
    ("API fetch data supabase",            ["supabase", "api", "fetch", "client"]),
    ("member subscription billing",       ["member", "subscription", "billing", "payment"]),
    ("push notification alert",           ["notification", "notify", "alert", "push"]),
    ("gym owner settings",                ["owner", "settings", "gym", "config"]),
    ("create account onboarding",         ["onboarding", "create", "register", "signup"]),
]


class ModelBench:
    def __init__(self, name: str, model_file: str, tokenizer_dir: str | None = None):
        self.name = name
        self.model_dir = MODEL_ROOT / name
        self.model_file = model_file
        self.tokenizer_dir = self.model_dir if tokenizer_dir is None else Path(tokenizer_dir)

    def model_size_mb(self) -> float:
        total = 0
        for f in self.model_dir.iterdir():
            if f.suffix in (".onnx", ".data") or f.name.endswith(".onnx_data"):
                total += f.stat().st_size
        return total / 1e6

    def load(self):
        t = time.perf_counter()
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_dir / "tokenizer.json"))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256 if "minilm" in self.name else 512)
        self.session = ort.InferenceSession(
            str(self.model_dir / self.model_file),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        # Probe dim
        probe = self._embed_raw(["x"])
        self.dim = probe.shape[1]
        return time.perf_counter() - t

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.session.run(None, feed)
        hidden = out[0]
        m = mask[:, :, np.newaxis].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / m.sum(axis=1).clip(min=1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return (pooled / norms).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed_raw(texts)


def run_index(bench: ModelBench, batch_size: int = 32) -> dict:
    from ccindex.config import Config
    from ccindex.walker import walk_project
    from ccindex.chunker import chunk_file
    from ccindex.index import Index

    cfg = Config()
    files = list(walk_project(FITIX, cfg))
    all_chunks = []
    for p in files:
        all_chunks.extend(chunk_file(p, FITIX, cfg))

    n_chunks = len(all_chunks)
    print(f"  {len(files)} files, {n_chunks} chunks")

    # Wipe existing index
    idx_dir = FITIX / ".ccindex"
    if idx_dir.exists():
        shutil.rmtree(idx_dir)

    sorted_chunks = sorted(all_chunks, key=lambda c: len(c.chunk_text))
    index = Index(FITIX / ".ccindex" / "index.db", embedding_dim=bench.dim)

    t_start = time.perf_counter()
    for i in range(0, len(sorted_chunks), batch_size):
        batch = sorted_chunks[i:i + batch_size]
        embeddings = bench.embed([c.chunk_text for c in batch])
        from ccindex.chunker import Chunk
        index.upsert_chunks(list(zip(batch, embeddings)))
        done = i + batch_size
        if done % 500 < batch_size:
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_chunks - done) / rate if rate > 0 else 0
            print(f"  {min(done,n_chunks):4d}/{n_chunks}  {elapsed:.0f}s elapsed  ~{eta:.0f}s remaining")

    index_time = time.perf_counter() - t_start
    print(f"  Index complete: {index_time:.1f}s = {index_time/60:.1f}min")

    return {
        "n_files": len(files),
        "n_chunks": n_chunks,
        "index_s": index_time,
        "index": index,
    }


def run_quality(bench: ModelBench, index, n_results: int = 5) -> dict:
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    query_times = []

    print(f"\n  Quality eval ({len(EVAL_QUERIES)} queries):")
    for query, expected_keywords in EVAL_QUERIES:
        t = time.perf_counter()
        q_vec = bench.embed([query])[0]
        results = index.vector_search(q_vec, top_k=n_results)
        qt = time.perf_counter() - t
        query_times.append(qt)

        # Check if any expected keyword appears in top-N file paths
        paths = [r.file_path.lower() for r in results]
        hit1 = any(kw in paths[0] for kw in expected_keywords) if paths else False
        hit3 = any(any(kw in p for kw in expected_keywords) for p in paths[:3])
        hit5 = any(any(kw in p for kw in expected_keywords) for p in paths[:5])
        hits_at_1 += hit1
        hits_at_3 += hit3
        hits_at_5 += hit5

        top_path = results[0].file_path if results else "—"
        top_score = results[0].score if results else 0
        marker = "✓" if hit1 else ("~" if hit3 else "✗")
        print(f"  {marker} '{query}'")
        print(f"    → {top_path} (score={top_score:.3f}) [{qt*1000:.0f}ms]")

    n = len(EVAL_QUERIES)
    return {
        "hits_at_1": hits_at_1 / n,
        "hits_at_3": hits_at_3 / n,
        "hits_at_5": hits_at_5 / n,
        "avg_query_ms": np.mean(query_times) * 1000,
    }


def main():
    models = [
        ModelBench("jina-code-onnx", "model-int8.onnx"),
        ModelBench("minilm-l6-onnx",  "model.onnx"),
    ]

    all_results = []

    for bench in models:
        print(f"\n{'='*65}")
        print(f"MODEL: {bench.name}/{bench.model_file}")
        print(f"  Size on disk: {bench.model_size_mb():.0f} MB")

        load_s = bench.load()
        print(f"  Load time: {load_s:.2f}s  |  Embedding dim: {bench.dim}")

        print(f"\n  Indexing fitix...")
        idx_result = run_index(bench)

        q_result = run_quality(bench, idx_result["index"])

        result = {
            "model": f"{bench.name}/{bench.model_file}",
            "size_mb": bench.model_size_mb(),
            "dim": bench.dim,
            "load_s": load_s,
            **idx_result,
            **q_result,
        }
        all_results.append(result)
        print(f"\n  Hits@1={q_result['hits_at_1']:.0%}  "
              f"Hits@3={q_result['hits_at_3']:.0%}  "
              f"Hits@5={q_result['hits_at_5']:.0%}  "
              f"AvgQuery={q_result['avg_query_ms']:.0f}ms")

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Model':<32} {'Size':>6} {'Dim':>4} {'Index':>8} {'Q@1':>5} {'Q@3':>5} {'Q@5':>5} {'QTime':>7}")
    print("-"*75)
    for r in all_results:
        print(f"{r['model']:<32} {r['size_mb']:>5.0f}M {r['dim']:>4} "
              f"{r['index_s']:>7.0f}s "
              f"{r['hits_at_1']:>5.0%} {r['hits_at_3']:>5.0%} {r['hits_at_5']:>5.0%} "
              f"{r['avg_query_ms']:>6.0f}ms")
    print("="*75)

    if len(all_results) == 2:
        a, b = all_results[0], all_results[1]
        speed_ratio = a["index_s"] / b["index_s"]
        quality_diff = b["hits_at_3"] - a["hits_at_3"]
        print(f"\nMiniLM vs Jina INT8:")
        print(f"  Speed:   {speed_ratio:.1f}x faster" if speed_ratio > 1 else f"  Speed:   {1/speed_ratio:.1f}x slower")
        print(f"  Quality: {quality_diff:+.0%} Hits@3 difference")

    # Save markdown results
    lines = [
        "# Model Benchmark Results",
        f"\nProject: fitix ({all_results[0]['n_files']} files, {all_results[0]['n_chunks']} chunks)",
        f"Hardware: Apple M4 Pro (macOS)",
        "\n## Summary\n",
        f"| Model | Size | Dim | Index Time | Hits@1 | Hits@3 | Hits@5 | Avg Query |",
        f"|-------|------|-----|-----------|--------|--------|--------|-----------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['model']} | {r['size_mb']:.0f}MB | {r['dim']} | "
            f"{r['index_s']:.0f}s ({r['index_s']/60:.1f}min) | "
            f"{r['hits_at_1']:.0%} | {r['hits_at_3']:.0%} | {r['hits_at_5']:.0%} | "
            f"{r['avg_query_ms']:.0f}ms |"
        )

    lines += [
        "\n## Eval Queries",
        "\n8 queries with keyword-in-path relevance check.",
        "✓ = hit in top-1, ~ = hit in top-3, ✗ = miss in top-5\n",
    ]

    RESULTS_FILE.write_text("\n".join(lines))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
