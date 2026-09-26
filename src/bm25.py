"""BM25 по трём полям объявления + RRF + учёт локации.

Для каждого поля (заголовок, параметры, описание) строится свой BM25-индекс:
так совпадение в коротком заголовке не "размывается" длинным описанием, а
каждое поле получает свою нормализацию на длину.

Выдачи полей объединяются Reciprocal Rank Fusion: score = sum 1 / (k + rank).
RRF работает с местами, а не со скорами, поэтому разный масштаб скоров BM25
по разным полям не мешает.

Локация. В train 75-83% выбранных объявлений находятся в той же локации, что и
запрос, а в городе запроса обычно ~1000 объявлений вместо 190k по всему корпусу.
Поэтому строим две выдачи:
  * локальную - BM25 только среди объявлений из локации запроса;
  * глобальную - BM25 по всему корпусу (для оставшихся ~25% и для запросов,
    в локации которых нет объявлений).
Итог: первые LOCAL_QUOTA мест - из локальной выдачи, остальное добивается
глобальной. Это ещё и решает проблему клонов: одинаковые объявления одного
исполнителя в разных городах имеют одинаковый BM25, и локальная выдача
выбирает копию из нужного города.

Запуск:
  uv run python -m src.bm25 val      # Recall@50 на локальной валидации
  uv run python -m src.bm25 submit   # answer.csv для бенчмарка
"""

import sys
from collections import defaultdict

import bm25s
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.validation import CACHE, corpus_mask, recall_at_k

FIELDS = ["title_norm", "params_norm", "desc_norm"]
TOP_K_FIELD = 200  # сколько кандидатов берём из каждого поля перед слиянием
TOP_K = 50         # итоговый размер ответа
RRF_K = 60         # сглаживающая константа RRF
LOCAL_QUOTA = 50   # сколько мест из 50 отдаём локальной выдаче (подобрано на валидации:
                   # Recall растёт монотонно, 0 -> 0.415, 50 -> 0.764)


def top_positive(scores: np.ndarray, k: int) -> np.ndarray:
    """Позиции k документов с наибольшим скором, только со скором > 0.

    Нулевой скор значит, что ни одно слово запроса не совпало, - это шум.
    argpartition находит топ-k за O(n) без полной сортировки 190k скоров.
    """
    if len(scores) > k:
        idx = np.argpartition(-scores, k)[:k]
    else:
        idx = np.arange(len(scores))
    idx = idx[scores[idx] > 0]
    return idx[np.argsort(-scores[idx])]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Сливает несколько ранжированных списков в один по Reciprocal Rank Fusion."""
    score = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] += 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)


def retrieve(items: pd.DataFrame, queries: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    """Для каждого запроса возвращает (локальную, глобальную) выдачу в виде item_id."""
    retrievers = {}
    for f in FIELDS:
        retrievers[f] = bm25s.BM25()  # k1=1.5, b=0.75 по умолчанию
        retrievers[f].index(items[f].str.split().tolist(), show_progress=False)

    item_ids = items.item_id.to_numpy()
    # позиции объявлений каждой локации - чтобы не маскировать 190k скоров на каждый запрос
    loc_positions = items.groupby("item_location_id").indices

    local_res, global_res = [], []
    for tokens, loc in tqdm(zip(queries.query_norm.str.split(), queries.search_location_id),
                            total=len(queries), desc="bm25"):
        if not tokens:  # запрос без токенов после нормализации ("1", одни стоп-слова)
            local_res.append([])
            global_res.append([])
            continue
        field_scores = [retrievers[f].get_scores(tokens) for f in FIELDS]

        # глобальная выдача: RRF полей по всему корпусу
        glob = rrf([top_positive(s, TOP_K_FIELD).tolist() for s in field_scores])

        # локальная: те же скоры, но только объявления из локации запроса
        pos = loc_positions.get(loc, np.array([], dtype=int))
        local = rrf([pos[top_positive(s[pos], TOP_K_FIELD)].tolist() for s in field_scores])

        local_res.append(item_ids[local].tolist())
        global_res.append(item_ids[glob].tolist())
    return local_res, global_res


def combine(local: list[str], glob: list[str], quota: int = LOCAL_QUOTA) -> list[str]:
    """Первые quota мест - локальная выдача, остальное - глобальная, без повторов.

    Если локальных кандидатов меньше quota, освободившиеся места тоже уходят глобальной.
    """
    result = list(dict.fromkeys(local[:quota]))
    seen = set(result)
    for item in glob:
        if len(result) >= TOP_K:
            break
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def main(mode: str) -> None:
    items = pd.read_parquet(CACHE / "items.parquet",
                            columns=["item_id", "in_benchmark", "item_location_id"] + FIELDS)

    if mode == "val":
        queries = pd.read_parquet(CACHE / "val_queries.parquet")
        items = items[corpus_mask(items, queries)].reset_index(drop=True)
        local, glob = retrieve(items, queries)
        # подбор квоты: 0 = только глобальная выдача (старый бейзлайн)
        for quota in [0, 10, 20, 30, 40, 45, 50]:
            preds = [combine(l, g, quota) for l, g in zip(local, glob)]
            print(f"quota={quota:2d}  Recall@50 {recall_at_k(preds, queries.relevant):.4f}")

    elif mode == "submit":
        queries = pd.read_parquet(CACHE / "queries.parquet")
        items = items[corpus_mask(items)].reset_index(drop=True)
        local, glob = retrieve(items, queries)
        preds = [combine(l, g) for l, g in zip(local, glob)]
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
