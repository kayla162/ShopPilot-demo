"""配對用的評分器：score(查詢, 候選) → 分數矩陣（查詢數 × 候選數），分數越高越像。

- BigramScorer：現有的字元雙字詞 Jaccard（transform/title.similarity），不需要任何模型。
- EmbeddingScorer：sentence-transformers 的多語言模型，向量正規化後算 cosine。第一次編碼時才 import、載入，
  載入後留在記憶體重複使用；沒裝套件、模型載入失敗都轉成 MatchError，不外洩例外內容。
"""
from __future__ import annotations

import threading

import numpy as np

from shoppilot.transform.title import similarity

MODEL_ALIASES = {"e5-small": "intfloat/multilingual-e5-small", "e5-base": "intfloat/multilingual-e5-base",
                 "bge-m3": "BAAI/bge-m3"}
MATCH_ERROR_LABELS = {
    "not_installed": "未安裝 sentence-transformers（pip install -r requirements-local.txt）",
    "model_load": "配對模型無法載入（模型名稱錯誤、還沒下載，或沒有網路）",
}
_LOADED: dict[tuple[str, str], object] = {}
_LOCK = threading.Lock()


class MatchError(Exception):
    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


class BigramScorer:
    name = "bigram"

    def score(self, queries: list[str], candidates: list[str]) -> np.ndarray:
        return np.array([[similarity(q, c) for c in candidates] for q in queries], dtype=float).reshape(
            len(queries), len(candidates))


class EmbeddingScorer:
    def __init__(self, model: str, device: str = "auto", name: str | None = None) -> None:
        self.model = model
        self.device = device
        self.name = name or model

    def _load(self):
        key = (self.model, self.device)
        with _LOCK:
            if key in _LOADED:
                return _LOADED[key]
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise MatchError("not_installed", str(e)) from e
            try:
                _LOADED[key] = SentenceTransformer(self.model, device=None if self.device == "auto" else self.device)
            except Exception as e:  # noqa: BLE001 — 模型名稱錯誤、沒下載、沒網路都算載入失敗
                raise MatchError("model_load", str(e)) from e
            return _LOADED[key]

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        # e5 系列要求查詢加「query: 」、文件加「passage: 」；其他模型不加
        return [f"{kind}: {t}" for t in texts] if "e5" in self.model.lower() else list(texts)

    def score(self, queries: list[str], candidates: list[str]) -> np.ndarray:
        model = self._load()
        if not queries or not candidates:
            return np.zeros((len(queries), len(candidates)))
        q = np.asarray(model.encode(self._prefix(queries, "query"), normalize_embeddings=True))
        c = np.asarray(model.encode(self._prefix(candidates, "passage"), normalize_embeddings=True))
        return q @ c.T


def make_scorer(name: str, device: str = "auto") -> BigramScorer | EmbeddingScorer:
    if name == "bigram":
        return BigramScorer()
    return EmbeddingScorer(MODEL_ALIASES.get(name, name), device=device, name=name)
