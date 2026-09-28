"""Анализ данных и ошибок, на котором основаны решения (см. README).

Запуск:
  uv run python -m src.analysis gap        # чем валидация отличается от бенчмарка
  uv run python -m src.analysis location   # сколько правильных ответов в локации запроса / рядом
  uv run python -m src.analysis clones     # Recall BM25 без локации в зависимости от числа клонов
  uv run python -m src.analysis misses     # почему правильные ответы не попадают в пул
                                           # (нужен cache/cands_eval.parquet от src.ranker pools)
"""

import sys

import numpy as np
import pandas as pd

from src.candidates import haversine_km, location_centroids
from src.validation import CACHE, corpus_mask


def gap() -> None:
    """Профиль запросов валидации и бенчмарка по признакам, известным для обоих."""
    items = pd.read_parquet(CACHE / "items.parquet", columns=["item_location_id", "in_benchmark"])
    bench_locs = items.item_location_id[items.in_benchmark]
    for name, file in [("val", "val_queries.parquet"), ("bench", "queries.parquet")]:
        d = pd.read_parquet(CACHE / file)
        ntok = d.query_norm.str.split().str.len()
        print(f"{name:6s} category=0: {(d.search_category == 0).mean():.3f}  "
              f"без фильтров: {(d.search_infm_params_text == '').mean():.3f}  "
              f"слов в запросе: {ntok.mean():.2f}  "
              f"локация есть в корпусе: {d.search_location_id.isin(bench_locs).mean():.3f}")


def location() -> None:
    """Где лежат правильные ответы относительно локации запроса."""
    items = pd.read_parquet(CACHE / "items.parquet", columns=[
        "item_id", "in_benchmark", "item_location_id", "item_latitude", "item_longitude"])
    val = pd.read_parquet(CACHE / "val_queries.parquet")
    corpus = items[corpus_mask(items, val)]
    loc_cnt = corpus.item_location_id.value_counts()
    print("объявлений корпуса в локации запроса:",
          val.search_location_id.map(loc_cnt).fillna(0).describe(percentiles=[.1, .5, .9]).round(0).to_dict())

    rel = val[["search_location_id", "relevant"]].explode("relevant").merge(
        items, left_on="relevant", right_on="item_id")
    same = (rel.item_location_id == rel.search_location_id).to_numpy()
    c = location_centroids(items).reindex(rel.search_location_id).to_numpy()
    dist = haversine_km(c[:, 0], c[:, 1], rel.item_latitude.to_numpy(), rel.item_longitude.to_numpy())
    print(f"правильный ответ в локации запроса: {same.mean():.3f}")
    print("расстояние до ответов из другой локации, км:",
          pd.Series(dist[~same]).describe(percentiles=[.25, .5, .75]).round(0).to_dict())


def clones() -> None:
    """Recall BM25 без учёта локации в зависимости от числа клонов правильного ответа.

    Клоны - объявления бенчмарка с таким же нормализованным заголовком.
    У клонов одинаковый BM25, и без локации правильная копия теряется среди них.
    """
    from src.bm25 import FIELDS, combine, retrieve

    items = pd.read_parquet(CACHE / "items.parquet",
                            columns=["item_id", "in_benchmark", "item_location_id"] + FIELDS)
    val = pd.read_parquet(CACHE / "val_queries.parquet")
    corpus = items[corpus_mask(items, val)].reset_index(drop=True)
    local, glob = retrieve(corpus, val)
    preds = [combine(l, g, quota=0) for l, g in zip(local, glob)]  # quota=0: без локации

    title_cnt = items[items.in_benchmark].title_norm.value_counts()
    titles = items.set_index("item_id").title_norm
    hits, n_clones = [], []
    for p, rel in zip(preds, val.relevant):
        for r in rel:
            hits.append(r in set(p))
            n_clones.append(title_cnt.get(titles[r], 0))
    d = pd.DataFrame({"hit": hits, "clones": pd.cut(n_clones, [-1, 0, 5, 50, np.inf],
                                                    labels=["0", "1-5", "6-50", ">50"])})
    print(d.groupby("clones", observed=True).hit.agg(recall="mean", n="size").round(3))
    print(f"доля объявлений бенчмарка с неуникальным заголовком: "
          f"{items[items.in_benchmark].title_norm.duplicated(keep=False).mean():.3f}")


def misses() -> None:
    """Раскладывает правильные ответы валидации, не попавшие в пул, по причинам."""
    val = pd.read_parquet(CACHE / "val_queries.parquet")
    n_rq = len(pd.read_parquet(CACHE / "ranker_queries.parquet", columns=["query_id"]))
    cands = pd.read_parquet(CACHE / "cands_eval.parquet", columns=["q", "item_id", "is_val"])
    cands = cands[cands.is_val]
    in_pool = set(zip(cands.q - n_rq, cands.item_id))
    items = pd.read_parquet(CACHE / "items.parquet", columns=[
        "item_id", "item_location_id", "title_norm", "params_norm", "desc_norm"]).set_index("item_id")

    reasons, examples = [], []
    for qi, (qn, sq, loc, rel) in enumerate(zip(val.query_norm, val.search_query,
                                                val.search_location_id, val.relevant)):
        q = set(qn.split())
        for r in rel:
            it = items.loc[r]
            if (qi, r) in in_pool:
                reasons.append("в пуле")
                continue
            if not q:
                reasons.append("пустой запрос")
                continue
            words = set(it.title_norm.split()) | set(it.params_norm.split()) | set(it.desc_norm.split())
            overlap = bool(q & words)
            reasons.append(("есть общие слова, " if overlap else "НЕТ общих слов, ")
                           + ("та же локация" if it.item_location_id == loc else "другая локация"))
            if not overlap:
                examples.append((sq, it.title_norm))
    print(pd.Series(reasons).value_counts(normalize=True).round(3))
    print("\nпримеры промахов без общих слов (запрос -> заголовок ответа):")
    for e in examples[:15]:
        print("  ", e)


if __name__ == "__main__":
    {"gap": gap, "location": location, "clones": clones, "misses": misses}[sys.argv[1]]()
