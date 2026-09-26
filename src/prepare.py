"""Подготовка данных: нормализация всех текстов один раз и сохранение в cache/.

Запуск: uv run python -m src.prepare

Результат:
  cache/items.parquet    - все известные объявления (benchmark_items + уникальные
                           объявления из train) с нормализованными полями.
                           Объявления из train нужны для псевдо-корпуса на валидации.
  cache/train.parquet    - пары запрос-объявление из train (только нужные колонки)
                           + нормализованный запрос.
  cache/queries.parquet  - запросы бенчмарка + нормализованный запрос.

Нормализация описаний - самая долгая часть, поэтому она распараллелена
по процессам и делается один раз, а все эксперименты читают готовый кэш.
"""

from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.text import normalize

DATA = Path("dataset")
CACHE = Path("cache")

ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_location_id",
    "item_latitude", "item_longitude", "item_price", "item_rating",
    "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden",
]
SEARCH_COLS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def normalize_many(texts: pd.Series, desc: str) -> list[str]:
    """Параллельная нормализация колонки текстов."""
    texts = texts.fillna("").tolist()
    with Pool() as pool:
        return list(tqdm(pool.imap(normalize, texts, chunksize=2000),
                         total=len(texts), desc=desc))


def main() -> None:
    CACHE.mkdir(exist_ok=True)

    train = pd.read_parquet(DATA / "train.parquet")
    bench_items = pd.read_parquet(DATA / "benchmark_items.parquet")
    queries = pd.read_parquet(DATA / "benchmark_queries.parquet")

    # --- объявления: корпус бенчмарка + всё, что встречалось в train ---
    # В train одно объявление повторяется во многих строках (разные запросы),
    # оставляем по одной строке на item_id.
    train_items = train[ITEM_COLS].drop_duplicates("item_id")
    bench_items = bench_items[ITEM_COLS].assign(in_benchmark=True)
    items = (pd.concat([bench_items, train_items.assign(in_benchmark=False)])
             .drop_duplicates("item_id", keep="first")  # benchmark-версия приоритетнее
             .reset_index(drop=True))
    # decimal-колонки из parquet неудобны в numpy, переводим во float
    for col in ["item_latitude", "item_longitude", "item_price"]:
        items[col] = items[col].astype(float)
    print(f"items: {len(items)} (в бенчмарке {items.in_benchmark.sum()})")

    items["title_norm"] = normalize_many(items.item_title_raw, "title")
    items["params_norm"] = normalize_many(items.item_infm_params_text, "params")
    items["desc_norm"] = normalize_many(items.item_description_raw, "description")
    items.to_parquet(CACHE / "items.parquet", index=False)

    # --- train: только пары и признаки запроса, тексты объявлений уже в items ---
    train = train[SEARCH_COLS + ["item_id", "item_location_id", "item_microcat_id"]].copy()
    uniq_q = pd.Series(train.search_query.unique())
    q_norm = dict(zip(uniq_q, normalize_many(uniq_q, "train queries")))
    train["query_norm"] = train.search_query.map(q_norm)
    train["params_norm"] = normalize_many(train.search_infm_params_text, "train params")
    train.to_parquet(CACHE / "train.parquet", index=False)

    # --- запросы бенчмарка ---
    queries["query_norm"] = normalize_many(queries.search_query, "bench queries")
    queries["params_norm"] = normalize_many(queries.search_infm_params_text, "bench params")
    queries.to_parquet(CACHE / "queries.parquet", index=False)


if __name__ == "__main__":
    main()
