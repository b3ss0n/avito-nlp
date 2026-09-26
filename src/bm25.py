"""Бейзлайн: BM25 по трём полям объявления + слияние через RRF.

Для каждого поля (заголовок, параметры, описание) строится свой BM25-индекс:
так совпадение в коротком заголовке не "размывается" длинным описанием, а
каждое поле получает свою нормализацию на длину.

Выдачи полей объединяются Reciprocal Rank Fusion: score = sum 1 / (k + rank).
RRF работает с местами, а не со скорами, поэтому разный масштаб скоров BM25
по разным полям не мешает.

Запуск:
  uv run python -m src.bm25 val      # Recall@50 на локальной валидации
  uv run python -m src.bm25 submit   # answer.csv для бенчмарка
"""

import sys
from collections import defaultdict
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from src.validation import CACHE, corpus_mask, recall_at_k

FIELDS = ["title_norm", "params_norm", "desc_norm"]
TOP_K_FIELD = 200  # сколько кандидатов берём из каждого поля перед слиянием
TOP_K = 50         # итоговый размер ответа
RRF_K = 60         # сглаживающая константа RRF


def bm25_search(docs: pd.Series, queries: list[list[str]], k: int) -> list[list[int]]:
    """Ищет по одному полю. Возвращает для каждого запроса позиции документов в docs.

    Документы с нулевым скором выкидываем: bm25s возвращает их, даже когда
    ни одно слово запроса не совпало, и это был бы случайный шум.
    """
    retriever = bm25s.BM25()  # k1=1.5, b=0.75 по умолчанию
    retriever.index(docs.str.split().tolist(), show_progress=False)
    idx, scores = retriever.retrieve(queries, k=k, show_progress=False)
    return [row_idx[row_sc > 0].tolist() for row_idx, row_sc in zip(idx, scores)]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Сливает несколько ранжированных списков в один по Reciprocal Rank Fusion."""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] += 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


def retrieve(items: pd.DataFrame, queries: pd.DataFrame) -> dict[str, list[list[str]]]:
    """Выдачи по каждому полю и итоговая после RRF, в виде списков item_id."""
    query_tokens = queries.query_norm.str.split().tolist()
    item_ids = items.item_id.to_numpy()

    per_field = {f: bm25_search(items[f], query_tokens, TOP_K_FIELD) for f in FIELDS}
    fused = [rrf([per_field[f][i] for f in FIELDS]) for i in range(len(queries))]

    result = {f: [item_ids[r].tolist() for r in rankings] for f, rankings in per_field.items()}
    result["rrf"] = [item_ids[r].tolist() for r in fused]
    return result


def main(mode: str) -> None:
    items = pd.read_parquet(CACHE / "items.parquet", columns=["item_id", "in_benchmark"] + FIELDS)

    if mode == "val":
        queries = pd.read_parquet(CACHE / "val_queries.parquet")
        items = items[corpus_mask(items, queries)].reset_index(drop=True)
        result = retrieve(items, queries)
        for name, preds in result.items():
            print(f"Recall@50 {name:12s} {recall_at_k(preds, queries.relevant):.4f}")

    elif mode == "submit":
        queries = pd.read_parquet(CACHE / "queries.parquet")
        items = items[corpus_mask(items)].reset_index(drop=True)
        preds = retrieve(items, queries)["rrf"]
        answer = pd.DataFrame({
            "query_id": queries.query_id,
            "answer": [" ".join(p[:TOP_K]) for p in preds],
        })
        answer.to_csv("answer.csv", index=False)
        check_answer("answer.csv", queries, items)
        print(f"answer.csv: {len(answer)} строк, пустых ответов: {(answer.answer == '').sum()}")


def check_answer(path: str, queries: pd.DataFrame, items: pd.DataFrame) -> None:
    """Проверяет answer.csv по требованиям из условия задачи."""
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(answer.columns) == ["query_id", "answer"], "ровно две колонки"
    assert len(answer) == len(queries) and set(answer.query_id) == set(queries.query_id), \
        "по строке на каждый query_id, без пропусков и повторов"
    valid_ids = set(items.item_id)
    for ids in answer.answer.str.split():
        assert len(ids) <= TOP_K, "не больше 50 item_id"
        assert len(set(ids)) == len(ids), "без повторов внутри строки"
        assert set(ids) <= valid_ids, "все item_id из benchmark_items"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
