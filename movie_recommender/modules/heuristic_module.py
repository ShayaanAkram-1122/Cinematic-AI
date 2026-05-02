"""Heuristic scoring and min-max normalization over a candidate set."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from modules.utils import genre_match_score, map_ui_genre_to_ml, parse_movielens_genres


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


class HeuristicScorer:
    def __init__(self, df: pd.DataFrame):
        self.norm_vote = _minmax(df["vote_average"])
        self.norm_pop = _minmax(df["popularity"])
        self.norm_rev = _minmax(df["revenue"])
        self.index = df.index

    def row_norms(self, idx) -> tuple[float, float, float, float]:
        """Return normalized vote, pop, revenue for a row index in original df."""
        return (
            float(self.norm_vote.loc[idx]),
            float(self.norm_pop.loc[idx]),
            float(self.norm_rev.loc[idx]),
            0.0,
        )

    def compute_score(self, movie: pd.Series, user_prefs: Dict[str, Any]) -> float:
        idx = movie.name
        nv, npv, nrev, _ = self.row_norms(idx)
        genres = movie["genres_list"] if isinstance(movie["genres_list"], list) else parse_movielens_genres(str(movie.get("genres", "")))
        sel = {map_ui_genre_to_ml(g) for g in (user_prefs.get("genres") or [])}
        gm = genre_match_score(genres, sel) if sel else 1.0
        return float(0.35 * nv + 0.25 * npv + 0.25 * gm + 0.15 * nrev)

    def score_breakdown(self, movie: pd.Series, user_prefs: Dict[str, Any]) -> Dict[str, float]:
        idx = movie.name
        nv, npv, nrev, _ = self.row_norms(idx)
        genres = movie["genres_list"] if isinstance(movie["genres_list"], list) else parse_movielens_genres(str(movie.get("genres", "")))
        sel = {map_ui_genre_to_ml(g) for g in (user_prefs.get("genres") or [])}
        gm = genre_match_score(genres, sel) if sel else 1.0
        total = float(0.35 * nv + 0.25 * npv + 0.25 * gm + 0.15 * nrev)
        return {
            "normalized_vote_avg": nv,
            "normalized_popularity": npv,
            "genre_match_ratio": gm,
            "normalized_revenue": nrev,
            "heuristic_total": total,
        }

    def rank_movies(self, df: pd.DataFrame, user_prefs: Dict[str, Any]) -> pd.DataFrame:
        if df.empty:
            return df.assign(heuristic_score=0.0)
        scorer = HeuristicScorer(df)
        scores = df.apply(lambda r: scorer.compute_score(r, user_prefs), axis=1)
        out = df.copy()
        out["heuristic_score"] = scores
        return out.sort_values("heuristic_score", ascending=False)


def build_astar_norm_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nv": _minmax(df["vote_average"]),
            "np": _minmax(df["popularity"]),
        },
        index=df.index,
    )
