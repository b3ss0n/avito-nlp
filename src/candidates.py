"""Генерация пула кандидатов и признаков для ранкера.

Первый этап (кандидатогенерация) собирает широкий пул: несколько сотен
объявлений на запрос из локальной (город запроса) и глобальной BM25-выдачи.
Правильный ответ попадает в такой пул заметно чаще, чем в топ-50.
Второй этап (src/ranker.py) - CatBoost выбирает из пула лучшие 50.

Для каждой пары (запрос, кандидат) считаем признаки: скоры BM25 по полям,
места в локальной и глобальной выдаче, совпадение локации, расстояние,
покрытие слов запроса заголовком, число клонов, рейтинг и т.п.
"""

import bm25s
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.bm25 import FIELDS, TOP_K_FIELD, rrf, top_positive

LOCAL_N = 300   # сколько кандидатов берём из локальной выдачи
GLOBAL_N = 100  # и сколько из глобальной
NEAR_N = 100    # и сколько из соседних локаций
NEAR_KM = 100   # радиус "соседних" локаций: медиана расстояния до правильных
                # ответов из чужой локации - 39 км, 75-й перцентиль - 149 км
NO_RANK = 10_000  # "место" для кандидата, которого нет в данной выдаче


def location_centroids(items: pd.DataFrame) -> pd.DataFrame:
    """Координаты локации = медиана координат её объявлений.

    У запроса есть только search_location_id без координат, поэтому
    расстояние "запрос - объявление" считаем от центра локации запроса.
    """
    return items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по поверхности Земли между точками (в градусах), км."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def generate(items: pd.DataFrame, queries: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Пул кандидатов с признаками: одна строка на пару (запрос, кандидат).

    items - корпус поиска (позиции в нём = номера документов в BM25-индексах).
    """
    retrievers = {}
    for f in FIELDS:
        retrievers[f] = bm25s.BM25()
        retrievers[f].index(items[f].str.split().tolist(), show_progress=False)
    loc_positions = items.groupby("item_location_id").indices
    title_tokens = items.title_norm.str.split().map(set).to_numpy()
    item_lat = items.item_latitude.to_numpy(dtype=float)
    item_lon = items.item_longitude.to_numpy(dtype=float)
    item_loc = items.item_location_id.to_numpy()
    near_cache = {}

    def near_positions(loc) -> np.ndarray:
        """Объявления в радиусе NEAR_KM от центра локации, кроме самой локации.

        Разбор промахов пула: 9% правильных ответов лексически находятся, но
        лежат в соседней локации (область, город-спутник), а глобальный топ-100
        до них не дотягивается - его забивают клоны со всей страны.
        """
        if loc not in near_cache:
            if loc in centroids.index:
                lat, lon = centroids.loc[loc]
                dist = haversine_km(lat, lon, item_lat, item_lon)
                near_cache[loc] = np.flatnonzero((dist <= NEAR_KM) & (item_loc != loc))
            else:  # в корпусе нет объявлений из этой локации - центр неизвестен
                near_cache[loc] = np.array([], dtype=int)
        return near_cache[loc]

    parts = []
    for qi, (tokens, loc) in enumerate(tqdm(
            zip(queries.query_norm.str.split(), queries.search_location_id),
            total=len(queries), desc="candidates")):
        if not tokens:
            continue
        field_scores = [retrievers[f].get_scores(tokens) for f in FIELDS]

        glob = rrf([top_positive(s, TOP_K_FIELD).tolist() for s in field_scores])
        pos = loc_positions.get(loc, np.array([], dtype=int))
        local = rrf([pos[top_positive(s[pos], TOP_K_FIELD)].tolist() for s in field_scores])

        npos = near_positions(loc)
        near = rrf([npos[top_positive(s[npos], TOP_K_FIELD)].tolist() for s in field_scores])

        local, glob, near = local[:LOCAL_N], glob[:GLOBAL_N], near[:NEAR_N]
        cand = np.array(list(dict.fromkeys(local + near + glob)), dtype=int)
        if len(cand) == 0:  # ни одного совпавшего слова ни в одном поле
            continue
        local_rank = {d: r for r, d in enumerate(local)}
        glob_rank = {d: r for r, d in enumerate(glob)}
        near_rank = {d: r for r, d in enumerate(near)}

        qset = set(tokens)
        df = pd.DataFrame({
            "q": qi,
            "pos": cand,
            "local_rank": [local_rank.get(d, NO_RANK) for d in cand],
            "global_rank": [glob_rank.get(d, NO_RANK) for d in cand],
            "near_rank": [near_rank.get(d, NO_RANK) for d in cand],
            "n_local": len(local),
            "n_near": len(near),
            "q_ntok": len(tokens),
            # доля слов запроса, которые есть в заголовке: 1.0 = все слова совпали
            "title_cover": [len(qset & title_tokens[d]) / len(qset) for d in cand],
        })
        for f, s in zip(FIELDS, field_scores):
            df[f"bm25_{f.removesuffix('_norm')}"] = s[cand]
        parts.append(df)

    cands = pd.concat(parts, ignore_index=True)

    # --- признаки объявления ---
    # Сначала компактная таблица признаков по всем объявлениям корпуса (без
    # длинных текстов), потом берём её строки по позициям кандидатов.
    # Индексировать сам items нельзя: копия описаний для ~1M строк не влезает в память.
    feats = pd.DataFrame({
        "item_id": items.item_id,
        "item_loc": items.item_location_id,
        "lat": items.item_latitude,
        "lon": items.item_longitude,
        "rating": items.item_rating,
        "reviews": items.item_rating_reviews_count,
        "log_price": np.log1p(items.item_price.clip(lower=0)),  # бывает цена -1
        "phone_hidden": items.item_is_phone_hidden.astype(int),
        "msg_forbidden": items.item_is_message_forbidden.astype(int),
        "title_len": items.title_norm.str.count(" ") + 1,
        "desc_len": items.desc_norm.str.count(" ") + 1,
        # сколько в корпусе объявлений с таким же заголовком (клоны одного исполнителя)
        "n_clones": items.title_norm.map(items.title_norm.value_counts()),
        "microcat": items.item_microcat_id,
    })
    it = feats.iloc[cands.pos.to_numpy()].reset_index(drop=True)
    q_loc = queries.search_location_id.to_numpy()[cands.q]
    it["same_loc"] = (it.item_loc.to_numpy() == q_loc).astype(int)
    qc = centroids.reindex(q_loc).to_numpy()
    it["dist_km"] = haversine_km(qc[:, 0], qc[:, 1], it.lat.to_numpy(), it.lon.to_numpy())
    return pd.concat([cands, it.drop(columns=["item_loc", "lat", "lon"])], axis=1)


def add_labels(cands: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """target = 1, если кандидат - правильный ответ запроса."""
    pairs = queries[["relevant"]].reset_index(drop=True).explode("relevant")
    pairs = pd.DataFrame({"q": pairs.index, "item_id": pairs.relevant}).assign(target=1)
    cands = cands.merge(pairs, on=["q", "item_id"], how="left")
    cands["target"] = cands.target.fillna(0).astype(int)
    return cands
