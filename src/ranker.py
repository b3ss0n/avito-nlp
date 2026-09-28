"""Второй этап: CatBoost выбирает 50 лучших из пула кандидатов.

Обучающие запросы для ранкера набираем из train_part тем же способом, что и
валидацию (одна сессия на текст), - валидационные запросы в обучение ранкера
не попадают. Таргет: выбрал ли пользователь объявление (1/0). Для каждого
запроса берём 50 кандидатов с наибольшей предсказанной вероятностью.

Порядок внутри 50 метрике не важен, но ранкер решает, КАКИЕ 50 из ~225
кандидатов попадут в ответ, - в этом и прирост относительно простого правила
"сначала всё локальное".

Генерация пула - самая долгая часть (~15 мин), а от признаков и параметров
модели она не зависит. Поэтому пул считается один раз и кэшируется:
  cache/cands_eval.parquet   - пул для обучающих запросов ранкера + валидации
                               (корпус: бенчмарк + правильные ответы обоих наборов)
  cache/cands_bench.parquet  - пул для запросов бенчмарка (корпус: только бенчмарк)
  cache/ranker_queries.parquet - обучающие запросы ранкера с разметкой
  cache/train_stats.parquet  - train_part без сессий обучающих запросов ранкера
                               (для статистик по train без утечки таргета)

Запуск:
  uv run python -m src.ranker pools    # посчитать и закэшировать пулы
  uv run python -m src.ranker val      # обучить на запросах ранкера, Recall@50 на валидации
  uv run python -m src.ranker submit   # обучить на запросах ранкера + валидации, answer.csv
"""

import sys

import pandas as pd
from catboost import CatBoostClassifier

from src.bm25 import TOP_K, check_answer
from src.candidates import add_labels, generate, location_centroids
from src.validation import CACHE, build_split, corpus_mask, recall_at_k

N_RANKER_QUERIES = 6000
FEATURES = [
    "local_rank", "global_rank", "n_local", "q_ntok", "title_cover",
    "bm25_title", "bm25_params", "bm25_desc",
    "same_loc", "dist_km", "rating", "reviews", "log_price",
    "phone_hidden", "msg_forbidden", "title_len", "desc_len", "n_clones", "microcat",
]
CAT_FEATURES = ["microcat"]


def build_pools() -> None:
    """Считает пулы кандидатов и сохраняет их в cache/."""
    items = pd.read_parquet(CACHE / "items.parquet")
    centroids = location_centroids(items)

    # обучающие запросы ранкера - из train_part, т.е. заведомо не из валидации
    train_part = pd.read_parquet(CACHE / "train_part.parquet")
    train_stats, rq = build_split(train_part, n_val=N_RANKER_QUERIES, seed=7)
    rq.to_parquet(CACHE / "ranker_queries.parquet", index=False)
    train_stats.to_parquet(CACHE / "train_stats.parquet", index=False)

    # один корпус для обучающих запросов и валидации
    val = pd.read_parquet(CACHE / "val_queries.parquet")
    queries = pd.concat([rq, val], ignore_index=True)
    corpus = items[corpus_mask(items, queries)].reset_index(drop=True)
    cands = add_labels(generate(corpus, queries, centroids), queries)
    cands["is_val"] = cands.q >= len(rq)
    cands.to_parquet(CACHE / "cands_eval.parquet", index=False)

    bench = pd.read_parquet(CACHE / "queries.parquet")
    corpus = items[corpus_mask(items)].reset_index(drop=True)
    generate(corpus, bench, centroids).to_parquet(CACHE / "cands_bench.parquet", index=False)


def train_model(cands: pd.DataFrame) -> CatBoostClassifier:
    # запросы, где правильного ответа нет в пуле, ничему не учат - убираем
    has_pos = cands.groupby("q").target.transform("max") == 1
    train = cands[has_pos]
    print(f"ranker train: {train.q.nunique()} запросов, {len(train)} пар, "
          f"доля позитивов {train.target.mean():.4f}")
    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6,
                               cat_features=CAT_FEATURES, verbose=100, thread_count=-1)
    model.fit(train[FEATURES], train.target)
    return model


def top50(cands: pd.DataFrame, model: CatBoostClassifier, n_queries: int) -> list[list[str]]:
    """Для каждого запроса - 50 item_id с наибольшей вероятностью."""
    cands = cands.assign(p=model.predict_proba(cands[FEATURES])[:, 1])
    cands = cands.sort_values(["q", "p"], ascending=[True, False])
    best = cands.groupby("q").head(TOP_K).groupby("q").item_id.agg(list)
    return [best.get(i, []) for i in range(n_queries)]


def main(mode: str) -> None:
    if mode == "pools":
        build_pools()
        return

    cands = pd.read_parquet(CACHE / "cands_eval.parquet")
    n_rq = len(pd.read_parquet(CACHE / "ranker_queries.parquet", columns=["query_id"]))

    if mode == "val":
        val = pd.read_parquet(CACHE / "val_queries.parquet")
        model = train_model(cands[~cands.is_val])
        val_cands = cands[cands.is_val].assign(q=lambda d: d.q - n_rq)
        preds = top50(val_cands, model, len(val))
        # потолок: Recall идеального ранкера = доля правильных ответов, попавших в пул
        pool = val_cands.groupby("q").item_id.agg(list)
        pools = [pool.get(i, []) for i in range(len(val))]
        print(f"потолок пула: {recall_at_k(pools, val.relevant, k=10**6):.4f}")
        print(f"Recall@50 ranker: {recall_at_k(preds, val.relevant):.4f}")
        print(pd.Series(model.get_feature_importance(), index=FEATURES).sort_values(ascending=False).round(1))

    elif mode == "submit":
        # для финальной модели валидация - тоже обучающие данные: пул уже посчитан
        model = train_model(cands)
        bench = pd.read_parquet(CACHE / "queries.parquet")
        preds = top50(pd.read_parquet(CACHE / "cands_bench.parquet"), model, len(bench))
        answer = pd.DataFrame({"query_id": bench.query_id, "answer": [" ".join(p) for p in preds]})
        answer.to_csv("answer.csv", index=False)
        items = pd.read_parquet(CACHE / "items.parquet", columns=["item_id", "in_benchmark"])
        check_answer("answer.csv", bench, items[items.in_benchmark])
        print(f"answer.csv: {len(answer)} строк, пустых ответов: {(answer.answer == '').sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val")
