"""Fuse CSP, search order, heuristics, clusters, and ANN; multi-user compromise scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from modules.heuristic_module import HeuristicScorer
from modules.ml_module import predict_ratings_batch
from modules.utils import genre_match_score, map_ui_genre_to_ml, parse_movielens_genres


def _order_rank(movie_ids: List[int]) -> Dict[int, int]:
    return {mid: i for i, mid in enumerate(movie_ids)}


def _genre_pref_score(row: pd.Series, prefs: Dict[str, Any]) -> float:
    sel = {map_ui_genre_to_ml(g) for g in (prefs.get("genres") or [])}
    genres = row["genres_list"] if isinstance(row["genres_list"], list) else parse_movielens_genres(str(row.get("genres", "")))
    return float(genre_match_score(genres, sel) if sel else 0.55)


def build_recommendations(
    filtered_df: pd.DataFrame,
    search_ordered_ids: List[int],
    user_prefs: Dict[str, Any],
    movies_clustered: pd.DataFrame,
    user_cluster_row: Optional[pd.Series],
    model: MLPRegressor,
    mlb: MultiLabelBinarizer,
    scaler: StandardScaler,
    user_id: int,
) -> pd.DataFrame:
    if filtered_df.empty:
        return pd.DataFrame()
    df = filtered_df.copy()
    if "cluster_id" not in df.columns and "cluster_id" in movies_clustered.columns:
        cmap = movies_clustered.set_index("movieId")["cluster_id"]
        df["cluster_id"] = df["movieId"].map(cmap).fillna(-1).astype(int)
    elif "cluster_id" not in df.columns:
        df["cluster_id"] = -1

    scorer = HeuristicScorer(df)
    hs = df.apply(lambda r: scorer.compute_score(r, user_prefs), axis=1)
    df["heuristic_score"] = hs

    mids = df["movieId"].to_numpy(dtype=np.int64)
    df["predicted_rating"] = predict_ratings_batch(mids, model, mlb, scaler, movies_clustered)
    df["pred_norm"] = (df["predicted_rating"] - 0.5) / 4.5

    uc = -1
    if user_cluster_row is not None and isinstance(user_cluster_row, pd.Series):
        uc = int(user_cluster_row.get("user_cluster", -1))
    if uc >= 0:
        cluster_mask = movies_clustered["cluster_id"] == uc
        popular_in_cluster = set(movies_clustered.loc[cluster_mask, "movieId"].astype(int).tolist())
        sim = df["movieId"].apply(lambda m: 1.0 if int(m) in popular_in_cluster else 0.35)
        df["cluster_similarity_score"] = sim
    else:
        df["cluster_similarity_score"] = 0.5

    order = _order_rank(search_ordered_ids)
    df["search_rank"] = df["movieId"].map(lambda m: order.get(int(m), len(order)))
    df["search_order_norm"] = 1.0 - (df["search_rank"] / (df["search_rank"].max() + 1.0))

    df["final_score"] = 0.4 * df["heuristic_score"] + 0.4 * df["pred_norm"] + 0.2 * df["cluster_similarity_score"]
    df["final_score"] = df["final_score"] + 0.02 * df["search_order_norm"]

    df["dominant_ai"] = df.apply(
        lambda r: _dominant_component(r["heuristic_score"], r["pred_norm"], r["cluster_similarity_score"]),
        axis=1,
    )
    return df.sort_values("final_score", ascending=False)


def _dominant_component(h: float, p: float, c: float) -> str:
    m = max(h, p, c)
    if m == h:
        return "A* / Heuristic"
    if m == p:
        return "ANN"
    return "K-Means"


def _minimax_zero_sum(values: List[float], depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    """Alpha–beta over a vector of leaf utilities (toy layer for demo)."""
    if depth == 0 or not values:
        return 0.0
    if maximizing:
        v = -1e9
        for i, val in enumerate(values[:6]):
            rest = [x for j, x in enumerate(values) if j != i]
            score = val + _minimax_zero_sum(rest, depth - 1, alpha, beta, False)
            v = max(v, score)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return v
    v = 1e9
    for i, val in enumerate(values[:6]):
        rest = [x for j, x in enumerate(values) if j != i]
        score = -val + _minimax_zero_sum(rest, depth - 1, alpha, beta, True)
        v = min(v, score)
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v


def minimax_recommendations(
    filtered_df: pd.DataFrame,
    prefs_a: Dict[str, Any],
    prefs_b: Dict[str, Any],
    depth: int = 3,
    branch: int = 24,
) -> pd.DataFrame:
    """
    Rank by collective satisfaction (genre scores sum) and attach a shallow zero-sum minimax
    score over genre-utility differences for ordering tie-breaks.
    """
    if filtered_df.empty:
        return filtered_df
    pool = filtered_df.head(min(branch, len(filtered_df))).copy()
    pool["score_a"] = pool.apply(lambda r: _genre_pref_score(r, prefs_a), axis=1)
    pool["score_b"] = pool.apply(lambda r: _genre_pref_score(r, prefs_b), axis=1)
    pool["collective"] = pool["score_a"] + pool["score_b"]
    diffs = (pool["score_a"] - pool["score_b"]).tolist()
    root = _minimax_zero_sum(diffs, max(1, depth - 1), -1e9, 1e9, True)
    pool["minimax_utility"] = pool["score_a"] - pool["score_b"] + (root / max(len(diffs), 1))
    return pool.sort_values(["collective", "minimax_utility"], ascending=[False, False])
