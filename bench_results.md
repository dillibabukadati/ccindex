# Model Benchmark Results

Project: fitix (388 files, 3991 chunks)
Hardware: Apple M4 Pro (macOS)

## Summary

| Model | Size | Dim | Index Time | Hits@1 | Hits@3 | Hits@5 | Avg Query |
|-------|------|-----|-----------|--------|--------|--------|-----------|
| jina-code-onnx/model-int8.onnx | 965MB | 768 | 171s (2.9min) | 62% | 75% | 88% | 10ms |
| minilm-l6-onnx/model.onnx | 92MB | 384 | 29s (0.5min) | 50% | 88% | 100% | 3ms |

## Eval Queries

8 queries with keyword-in-path relevance check.
✓ = hit in top-1, ~ = hit in top-3, ✗ = miss in top-5
