"""埋め込み生成の境界と、ローカル Ollama API アダプタ。"""

from __future__ import annotations

import math
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict


class EmbeddingError(RuntimeError):
    """埋め込み応答を安全に利用できない。"""


class Embedder(Protocol):
    """dense 検索が必要とする最小契約。"""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _EmbedResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    embeddings: list[list[float]]


class OllamaEmbedder:
    """Ollama のローカル ``POST /api/embed`` を呼ぶ同期アダプタ。

    既定値はローカル URL のみで、チケットやポリシーを外部の有料 API へ送らない。
    モデルのダウンロードと起動は明示的な運用手順とし、通常の BM25 起動経路では
    このクラスを生成しない。
    """

    def __init__(
        self,
        *,
        model: str = "embeddinggemma",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)

    @property
    def model_id(self) -> str:
        return f"ollama:{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts, "truncate": False},
            )
            response.raise_for_status()
            payload = _EmbedResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

        if payload.model != self._model:
            raise EmbeddingError(
                f"embedding model mismatch: requested {self._model}, got {payload.model}"
            )
        if len(payload.embeddings) != len(texts):
            raise EmbeddingError(
                f"embedding count mismatch: expected {len(texts)}, got {len(payload.embeddings)}"
            )
        dimensions = {len(vector) for vector in payload.embeddings}
        if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) == 0:
            raise EmbeddingError("embeddings must have one consistent, non-zero dimension")
        if any(not math.isfinite(value) for vector in payload.embeddings for value in vector):
            raise EmbeddingError("embedding contains a non-finite value")
        return payload.embeddings
