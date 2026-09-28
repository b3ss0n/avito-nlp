"""Dense-эмбеддинги: признак смысловой близости запроса и заголовка для ранкера.

BM25 видит только совпадения слов. Эмбеддинг-модель переводит текст в вектор
так, что близкие по смыслу тексты оказываются рядом ("скупка телевизоров" и
"куплю ваш телевизор"). Здесь эмбеддинги используются как ПРИЗНАК ранкера:
косинус между запросом и заголовком каждого кандидата в пуле. Новых
кандидатов они не добавляют (разбор промахов в src/analysis.py показал, что
промахов без общих слов с запросом <5%, и это чаще шум, чем перефразировки).

Модель: intfloat/multilingual-e5-small (MIT, 384-мерные векторы), скачивается
один раз scripts/download_model.py и дальше работает офлайн. e5 обучена с
префиксами: "query: " для запросов и "passage: " для документов.

Кодируем только объявления, которые встречаются в пулах, - не весь корпус.
На CPU ~150 заголовков/с.

Запуск (после src.ranker pools):
  uv run python -m src.embeddings
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # никаких обращений к сети, только локальные веса

import numpy as np
import pandas as pd

from src.validation import CACHE

MODEL_DIR = "models/multilingual-e5-small"


def encode(model, texts: list[str], prefix: str) -> np.ndarray:
    # normalize_embeddings=True: векторы единичной длины, косинус = скалярное произведение
    return model.encode([prefix + t for t in texts], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)


def main() -> None:
    # импорт здесь, а не наверху: ранкеру для признака (add_emb_feature) torch не нужен
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_DIR, device="cpu")

    # запросы: в том же порядке, что q в пулах
    rq = pd.read_parquet(CACHE / "ranker_queries.parquet", columns=["search_query"])
    val = pd.read_parquet(CACHE / "val_queries.parquet", columns=["search_query"])
    bench = pd.read_parquet(CACHE / "queries.parquet", columns=["search_query"])
    np.save(CACHE / "emb_q_eval.npy", encode(model, pd.concat([rq, val]).search_query.tolist(), "query: "))
    np.save(CACHE / "emb_q_bench.npy", encode(model, bench.search_query.tolist(), "query: "))

    # объявления: только встречающиеся в пулах
    ids = pd.concat([pd.read_parquet(CACHE / f, columns=["item_id"]).item_id
                     for f in ["cands_eval.parquet", "cands_bench.parquet"]]).unique()
    items = pd.read_parquet(CACHE / "items.parquet", columns=["item_id", "item_title_raw"])
    items = items[items.item_id.isin(ids)].reset_index(drop=True)
    print(f"объявлений для кодирования: {len(items)}")
    np.save(CACHE / "emb_items.npy", encode(model, items.item_title_raw.fillna("").tolist(), "passage: "))
    items[["item_id"]].to_parquet(CACHE / "emb_item_ids.parquet", index=False)


def add_emb_feature(cands: pd.DataFrame, q_emb_file: str) -> pd.DataFrame:
    """emb_cos - косинус между эмбеддингами запроса и заголовка кандидата."""
    q_emb = np.load(CACHE / q_emb_file)
    item_emb = np.load(CACHE / "emb_items.npy")
    row = pd.Series(np.arange(len(item_emb)),
                    index=pd.read_parquet(CACHE / "emb_item_ids.parquet").item_id)
    it = row.reindex(cands.item_id).to_numpy()
    q = cands.q.to_numpy()
    # построчное скалярное произведение пар (запрос, кандидат), батчами ради памяти
    cos = np.empty(len(cands), dtype=np.float32)
    for s in range(0, len(cands), 1_000_000):
        e = slice(s, s + 1_000_000)
        cos[e] = np.einsum("ij,ij->i", q_emb[q[e]], item_emb[it[e]])
    return cands.assign(emb_cos=cos)


if __name__ == "__main__":
    main()
