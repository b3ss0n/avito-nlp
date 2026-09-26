"""Локальная валидация, повторяющая устройство бенчмарка.

Что выяснили в EDA (см. notebooks):
  * "запрос" бенчмарка = одна поисковая сессия: текст + локация + фильтры +
    флаг доставки + категория. В train такая сессия - группа строк с одинаковым
    ключом SESSION_KEY, а выбранные в ней объявления - её правильные ответы.
  * в бенчмарке все тексты запросов уникальны (одна сессия на текст);
  * 37% текстов бенчмарка встречаются в train.

Если выбрать текст равновероятно из train и взять одну его сессию, то доля
текстов, у которых в train остаются другие сессии, получается ~37%, т.е. ровно
как в бенчмарке. Поэтому валидацию строим именно так, без ручной подгонки.

Разбиение групповое: сессия целиком уходит либо в обучение, либо в валидацию
(аналог GroupShuffleSplit), иначе правильные ответы протекли бы в обучение.

Корпус для поиска на валидации = объявления бенчмарка (реалистичные
"конкуренты", 189k) + правильные объявления валидации (иначе их просто нет
в корпусе и Recall был бы ~0 для любого метода).

Запуск: uv run python -m src.validation
"""

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("cache")

SESSION_KEY = [
    "search_query", "search_location_id", "search_infm_params_text",
    "search_is_delivery_search", "search_category",
]


def build_split(train: pd.DataFrame, n_val: int = 3000, seed: int = 42):
    """Возвращает (train_part, val_queries).

    val_queries: одна строка на сессию, колонки как у benchmark_queries
    + query_id + relevant (список правильных item_id без повторов).
    """
    rng = np.random.default_rng(seed)
    train = train.copy()
    # id сессии - номер группы по ключу
    train["session_id"] = train.groupby(SESSION_KEY, dropna=False).ngroup()

    # 1) равновероятно выбираем тексты запросов
    texts = train.search_query.unique()
    val_texts = rng.choice(texts, size=n_val, replace=False)

    # 2) для каждого текста - одна случайная его сессия
    sessions = train.loc[train.search_query.isin(val_texts), ["search_query", "session_id"]]
    sessions = sessions.drop_duplicates()
    val_sessions = (sessions.sample(frac=1, random_state=seed)
                    .drop_duplicates("search_query")  # одна сессия на текст
                    .session_id)

    is_val = train.session_id.isin(val_sessions)
    val_rows = train[is_val]
    train_part = train[~is_val].drop(columns="session_id").reset_index(drop=True)

    # 3) сессия -> одна строка с признаками запроса и списком правильных ответов
    val_queries = (val_rows
                   .groupby("session_id")
                   .agg(**{c: (c, "first") for c in SESSION_KEY + ["query_norm", "params_norm"]},
                        relevant=("item_id", lambda s: sorted(set(s))))
                   .reset_index(drop=True))
    val_queries.insert(0, "query_id", [f"val_{i:05d}" for i in range(len(val_queries))])
    return train_part, val_queries


def corpus_mask(items: pd.DataFrame, val_queries: pd.DataFrame | None = None) -> np.ndarray:
    """Какие объявления участвуют в поиске.

    Для бенчмарка - только benchmark_items, для валидации - ещё и правильные
    объявления валидационных запросов.
    """
    mask = items.in_benchmark.to_numpy().copy()
    if val_queries is not None:
        val_items = set(val_queries.relevant.explode())
        mask |= items.item_id.isin(val_items).to_numpy()
    return mask


def recall_at_k(predictions: list[list[str]], relevant: list[list[str]], k: int = 50) -> float:
    """Recall@k как в условии: доля правильных объявлений в топ-k, среднее по запросам."""
    scores = []
    for pred, rel in zip(predictions, relevant):
        rel = set(rel)
        scores.append(len(rel & set(pred[:k])) / len(rel))
    return float(np.mean(scores))


def main() -> None:
    train = pd.read_parquet(CACHE / "train.parquet")
    train_part, val_queries = build_split(train)
    train_part.to_parquet(CACHE / "train_part.parquet", index=False)
    val_queries.to_parquet(CACHE / "val_queries.parquet", index=False)

    # --- проверяем, что валидация похожа на бенчмарк ---
    seen_text = val_queries.search_query.isin(train_part.search_query).mean()
    val_items = val_queries.relevant.explode()
    seen_item = val_items.isin(train_part.item_id).mean()
    items = pd.read_parquet(CACHE / "items.parquet", columns=["item_id", "in_benchmark"])
    in_bench = val_items.isin(items.item_id[items.in_benchmark]).mean()
    print(f"val sessions: {len(val_queries)}, train_part rows: {len(train_part)}")
    print(f"правильных объявлений на запрос: {val_queries.relevant.str.len().mean():.2f}")
    print(f"доля знакомых текстов: {seen_text:.3f} (в бенчмарке 0.370)")
    print(f"доля правильных объявлений, встречавшихся в train_part: {seen_item:.3f}")
    print(f"доля правильных объявлений, лежащих в benchmark_items: {in_bench:.3f}")
    print(f"размер корпуса валидации: {corpus_mask(items, val_queries).sum()}")


if __name__ == "__main__":
    main()
