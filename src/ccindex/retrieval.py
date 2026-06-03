from __future__ import annotations
import numpy as np
from ccindex.config import Config
from ccindex.index import Index, SearchResult
from ccindex.models import EmbeddingModel, Reranker

_CHARS_PER_TOKEN = 4


class Retriever:
    def __init__(
        self,
        index: Index,
        model: EmbeddingModel,
        reranker: Reranker,
        config: Config,
    ):
        self._index = index
        self._model = model
        self._reranker = reranker
        self._config = config

    def query(self, text: str) -> list[SearchResult]:
        query_vec = self._model.embed([text])[0]

        vec_results = self._index.vector_search(query_vec, top_k=50)
        fts_results = self._index.fts_search(text, top_k=20)
        candidates = self._dedupe(vec_results + fts_results)

        if not candidates:
            return []

        passages = [r.chunk_text for r in candidates]
        scores = self._reranker.rerank(text, passages)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        filtered = [
            result for result, score in ranked
            if score >= self._config.relevance_threshold
        ]

        capped = self._apply_token_cap(filtered)
        return capped[:self._config.top_k]

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        seen: set[int] = set()
        out: list[SearchResult] = []
        for r in results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                out.append(r)
        return out

    def _apply_token_cap(self, results: list[SearchResult]) -> list[SearchResult]:
        cap_chars = self._config.token_cap * _CHARS_PER_TOKEN
        used = 0
        out: list[SearchResult] = []
        for r in results:
            size = len(r.chunk_text)
            if used + size > cap_chars:
                break
            out.append(r)
            used += size
        return out


def format_hook_output(results: list[SearchResult]) -> str:
    if not results:
        return ""

    lines = ["[ccindex context]"]
    for r in results:
        loc = f"{r.file_path}"
        if r.start_line and r.end_line:
            loc += f":{r.start_line}-{r.end_line}"
        if r.symbol:
            loc += f" ({r.symbol})"
        lines.append(f"── {loc}")
        for line in r.chunk_text.splitlines()[:20]:
            lines.append(f"   {line}")
        lines.append("")
    lines.append("[end ccindex context]")
    return "\n".join(lines)
