"""Constraint Satisfaction Problem style filters on the movie catalog."""

from __future__ import annotations

from typing import Any, Dict, Set

import pandas as pd

from modules.utils import map_ui_genre_to_ml, parse_movielens_genres


class CSPFilter:
    """AND-composable constraints over a movie DataFrame."""

    @staticmethod
    def genre_constraint(df: pd.DataFrame, selected_genres: list[str]) -> pd.Series:
        if not selected_genres:
            return pd.Series(True, index=df.index)
        want: Set[str] = {map_ui_genre_to_ml(g) for g in selected_genres}

        def ok(row) -> bool:
            glist = row["genres_list"] if isinstance(row["genres_list"], list) else parse_movielens_genres(str(row.get("genres", "")))
            return bool(set(glist) & want)

        return df.apply(ok, axis=1)

    @staticmethod
    def year_constraint(df: pd.DataFrame, min_year: float, max_year: float) -> pd.Series:
        y = pd.to_numeric(df["year"], errors="coerce")
        return y.between(min_year, max_year, inclusive="both")

    @staticmethod
    def runtime_constraint(df: pd.DataFrame, min_rt: float, max_rt: float) -> pd.Series:
        r = pd.to_numeric(df["runtime"], errors="coerce").fillna(0)
        return r.between(min_rt, max_rt, inclusive="both")

    @staticmethod
    def rating_constraint(df: pd.DataFrame, min_rating: float) -> pd.Series:
        return pd.to_numeric(df["vote_average"], errors="coerce").fillna(0) >= float(min_rating)

    @staticmethod
    def popularity_constraint(df: pd.DataFrame, min_popularity: float) -> pd.Series:
        return pd.to_numeric(df["popularity"], errors="coerce").fillna(0) >= float(min_popularity)

    @staticmethod
    def apply_all_constraints(df: pd.DataFrame, user_prefs: Dict[str, Any]) -> pd.DataFrame:
        m = pd.Series(True, index=df.index)
        genres = user_prefs.get("genres") or []
        if genres:
            m &= CSPFilter.genre_constraint(df, genres)
        m &= CSPFilter.year_constraint(df, user_prefs["year_min"], user_prefs["year_max"])
        m &= CSPFilter.runtime_constraint(df, user_prefs["runtime_min"], user_prefs["runtime_max"])
        m &= CSPFilter.rating_constraint(df, user_prefs["min_rating"])
        m &= CSPFilter.popularity_constraint(df, user_prefs["min_popularity"])
        return df.loc[m].copy()


def constraint_filter_counts(df: pd.DataFrame, user_prefs: Dict[str, Any]) -> pd.DataFrame:
    """How many rows each constraint removes (for analytics)."""
    n0 = len(df)
    rows = []
    genres = user_prefs.get("genres") or []
    if genres:
        g_ok = CSPFilter.genre_constraint(df, genres)
        rows.append(("Genre (≥1 match)", int(g_ok.sum()), int((~g_ok).sum())))
    y_ok = CSPFilter.year_constraint(df, user_prefs["year_min"], user_prefs["year_max"])
    rows.append(("Year range", int(y_ok.sum()), int((~y_ok).sum())))
    r_ok = CSPFilter.runtime_constraint(df, user_prefs["runtime_min"], user_prefs["runtime_max"])
    rows.append(("Runtime range", int(r_ok.sum()), int((~r_ok).sum())))
    ra_ok = CSPFilter.rating_constraint(df, user_prefs["min_rating"])
    rows.append(("Min vote_average", int(ra_ok.sum()), int((~ra_ok).sum())))
    p_ok = CSPFilter.popularity_constraint(df, user_prefs["min_popularity"])
    rows.append(("Min popularity", int(p_ok.sum()), int((~p_ok).sum())))
    final = CSPFilter.apply_all_constraints(df, user_prefs)
    rows.append(("Combined (AND)", len(final), n0 - len(final)))
    return pd.DataFrame(rows, columns=["constraint", "passed", "filtered_out"])
