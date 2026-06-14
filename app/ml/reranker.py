from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Признаки кандидата: косинусная схожесть с лайком, 80 числовых фичей датасета,
# 3 флага совпадения категорий кандидата и последнего лайкнутого айтема.
# Категориальные признаки: cat_1..3 кандидата + cat_1..3 лайкнутого (кросс-фичи).
# Итого 90 признаков, 6 категориальных (индексы в _CAT_FEAT_IDX).
FLOAT_COLS = [f"float_{i}" for i in range(1, 81)]
ORIG_CAT_COLS = ["cat_1", "cat_2", "cat_3"]
LIKED_CAT_COLS = ["liked_cat_1", "liked_cat_2", "liked_cat_3"]
MATCH_COLS = ["cat_1_match", "cat_2_match", "cat_3_match"]
CAT_COLS = ORIG_CAT_COLS + LIKED_CAT_COLS
NUM_COLS = ["sim_to_liked"] + FLOAT_COLS + MATCH_COLS
FEATURE_COLS = NUM_COLS + CAT_COLS
_CAT_FEAT_IDX = [FEATURE_COLS.index(c) for c in CAT_COLS]


class SessionReranker:
    """CatBoost YetiRank-реранкер с учётом категорий лайкнутого айтема."""

    def __init__(self, model_path: str | None) -> None:
        """Загружает модель из .cbm-файла. При ошибке логирует предупреждение и остаётся недоступным."""
        self._model = None
        if model_path:
            try:
                from catboost import CatBoostRanker
                m = CatBoostRanker()
                m.load_model(model_path)
                self._model = m
            except Exception as e:
                logger.warning("Failed to load reranker model from %s: %s", model_path, e)

    @property
    def available(self) -> bool:
        return self._model is not None

    def _build_pool(
        self,
        sims: np.ndarray,
        float_mat: np.ndarray,
        cat_mat: np.ndarray,
        liked_cats: np.ndarray,
        match_mat: np.ndarray | None = None,
    ):
        from catboost import Pool

        n = len(sims)
        liked_broadcast = np.broadcast_to(liked_cats, (n, 3))
        # match_mat можно передать снаружи (напр., union по всем лайкам)
        if match_mat is None:
            match_mat = (cat_mat.astype(str) == liked_broadcast.astype(str)).astype(np.float32)
        liked_cat_mat = liked_broadcast.astype(object)

        X = np.empty((n, len(FEATURE_COLS)), dtype=object)
        X[:, 0] = sims
        X[:, 1:1 + len(FLOAT_COLS)] = float_mat
        X[:, 1 + len(FLOAT_COLS):len(NUM_COLS)] = match_mat
        X[:, len(NUM_COLS):len(NUM_COLS) + 3] = cat_mat
        X[:, len(NUM_COLS) + 3:] = liked_cat_mat
        return Pool(X, cat_features=_CAT_FEAT_IDX)

    def score_items(
        self,
        sims: np.ndarray,
        float_mat: np.ndarray,
        cat_mat: np.ndarray,
        liked_cats: np.ndarray,
        match_mat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Возвращает сырые скоры модели (n,) без сортировки."""
        pool = self._build_pool(sims, float_mat, cat_mat, liked_cats, match_mat)
        return self._model.predict(pool)

    def rerank(
        self,
        item_ids: list[int],
        sims: np.ndarray,
        float_mat: np.ndarray,
        cat_mat: np.ndarray,
        liked_cats: np.ndarray,
        match_mat: np.ndarray | None = None,
    ) -> list[int]:
        """Возвращает item_ids, отсортированные по убыванию скора модели.

        Кросс-фичи строятся здесь: флаги совпадения категорий (cat_*_match) и
        категории лайкнутого айтема (liked_cat_*), транслированные на всех кандидатов.
        match_mat можно передать снаружи — например, union-матч по всем лайкнутым айтемам.
        """
        if not self.available or len(item_ids) == 0:
            return item_ids
        scores = self.score_items(sims, float_mat, cat_mat, liked_cats, match_mat)
        order = np.argsort(scores)[::-1]
        return [item_ids[i] for i in order]
