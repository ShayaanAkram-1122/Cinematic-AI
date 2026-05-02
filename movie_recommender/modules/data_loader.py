"""Load and merge MovieLens, TMDB-style metadata, and Wikipedia plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from modules.utils import (
    extract_year_from_title,
    normalize_title,
    parse_movielens_genres,
)


def _ai_project_root() -> Path:
    # .../movie_recommender/modules/data_loader.py -> parents[2] == AI Project
    return Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    return _ai_project_root() / "Dataset"


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected dataset file missing: {path}. "
            f"Place CSVs under {dataset_dir()} (ml-32m/, top_rated_movies.csv, wiki_movie_plots_deduped.csv)."
        )


@st.cache_data(show_spinner="Loading and merging movie datasets…")
def load_merged_movies_and_ratings(
    ratings_sample_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens ml-32m, TMDB proxy (top_rated_movies.csv), and Wikipedia plots.
    If ratings_sample_fraction is in (0,1), randomly sample that fraction of rating rows (memory guard).
    """
    root = dataset_dir()
    ml = root / "ml-32m"
    p_movies = ml / "movies.csv"
    p_links = ml / "links.csv"
    p_ratings = ml / "ratings.csv"
    p_tmdb = root / "top_rated_movies.csv"
    p_wiki = root / "wiki_movie_plots_deduped.csv"
    for p in (p_movies, p_links, p_ratings, p_tmdb, p_wiki):
        _require_file(p)

    movies = pd.read_csv(p_movies)
    links = pd.read_csv(p_links)
    tmdb = pd.read_csv(p_tmdb)
    wiki = pd.read_csv(p_wiki)

    if "Unnamed: 0" in tmdb.columns:
        tmdb = tmdb.drop(columns=["Unnamed: 0"])

    wiki = wiki.rename(
        columns={
            "Title": "wiki_title",
            "Release Year": "wiki_release_year",
            "Director": "wiki_director",
            "Cast": "wiki_cast",
            "Genre": "wiki_genre",
            "Plot": "plot_summary",
        }
    )
    wiki["wiki_year"] = pd.to_numeric(wiki["wiki_release_year"], errors="coerce")
    wiki["norm_title"] = wiki["wiki_title"].astype(str).map(normalize_title)
    wiki_first = (
        wiki.sort_values("wiki_year")
        .drop_duplicates(subset=["norm_title"], keep="first")[
            ["norm_title", "wiki_year", "wiki_director", "wiki_cast", "wiki_genre", "plot_summary"]
        ]
    )

    tmdb = tmdb.rename(columns={"id": "tmdb_join_id"})
    tmdb = tmdb.drop_duplicates(subset=["tmdb_join_id"], keep="first")
    tmdb["year_tmdb"] = pd.to_datetime(tmdb.get("release_date"), errors="coerce").dt.year
    # Avoid column collisions with MovieLens `title` (pandas would drop/rename).
    tmdb = tmdb.drop(columns=[c for c in ("title",) if c in tmdb.columns])

    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")

    merged = movies.merge(links, on="movieId", how="left")
    merged = merged.merge(
        tmdb,
        left_on="tmdbId",
        right_on="tmdb_join_id",
        how="left",
    )

    merged["title_ml"] = merged["title"].astype(str)
    merged["year"] = merged["year_tmdb"]
    merged["year"] = merged["year"].fillna(merged["title_ml"].map(extract_year_from_title))
    merged["norm_title"] = merged["title_ml"].map(normalize_title)

    merged["genres_list"] = merged["genres"].astype(str).map(parse_movielens_genres)
    merged["genres"] = merged["genres_list"].apply(lambda g: "|".join(g) if g else "")

    def _col(name: str, default):
        return merged[name] if name in merged.columns else pd.Series(default, index=merged.index)

    merged["overview"] = _col("overview", "").fillna("").astype(str)
    merged["vote_average"] = pd.to_numeric(_col("vote_average", 0), errors="coerce").fillna(0.0)
    merged["vote_count"] = pd.to_numeric(_col("vote_count", 0), errors="coerce").fillna(0).astype(int)
    merged["popularity"] = pd.to_numeric(_col("popularity", 0), errors="coerce").fillna(0.0)
    merged["runtime"] = pd.to_numeric(_col("runtime", np.nan), errors="coerce").fillna(100.0)
    merged["budget"] = pd.to_numeric(_col("budget", 0), errors="coerce").fillna(0.0)
    merged["revenue"] = pd.to_numeric(_col("revenue", 0), errors="coerce").fillna(0.0)
    merged["keywords"] = _col("keywords", "").fillna("").astype(str)
    merged["director"] = _col("director", "").fillna("").astype(str)
    merged["cast"] = _col("cast", "").fillna("").astype(str)

    merged = merged.merge(wiki_first, on="norm_title", how="left")

    y = pd.to_numeric(merged["year"], errors="coerce")
    wy = pd.to_numeric(merged["wiki_year"], errors="coerce")
    mismatch = wy.notna() & y.notna() & (abs(y - wy) > 1)
    merged.loc[mismatch, ["plot_summary", "wiki_director", "wiki_cast", "wiki_genre", "wiki_year"]] = np.nan

    merged["director"] = np.where(
        merged["director"].astype(str).str.len() > 0,
        merged["director"],
        merged["wiki_director"].fillna(""),
    ).astype(str)
    merged["cast"] = np.where(
        merged["cast"].astype(str).str.len() > 0,
        merged["cast"],
        merged["wiki_cast"].fillna(""),
    ).astype(str)
    merged["plot_summary"] = merged["plot_summary"].fillna("").astype(str)
    short_plot = merged["plot_summary"].str.len() < 12
    merged.loc[short_plot, "plot_summary"] = merged.loc[short_plot, "overview"]

    out = pd.DataFrame(
        {
            "movieId": merged["movieId"].astype(int),
            "title": merged["title_ml"].astype(str),
            "genres": merged["genres"].astype(str),
            "genres_list": merged["genres_list"],
            "year": pd.to_numeric(merged["year"], errors="coerce"),
            "runtime": merged["runtime"].astype(float),
            "vote_average": merged["vote_average"].astype(float),
            "vote_count": merged["vote_count"].astype(int),
            "popularity": merged["popularity"].astype(float),
            "overview": merged["overview"].astype(str),
            "plot_summary": merged["plot_summary"].astype(str),
            "director": merged["director"].astype(str),
            "cast": merged["cast"].astype(str),
            "budget": merged["budget"].astype(float),
            "revenue": merged["revenue"].astype(float),
            "keywords": merged["keywords"].astype(str),
        }
    )
    out["year"] = out["year"].fillna(out["title"].map(extract_year_from_title))
    out = out.drop_duplicates(subset=["movieId"], keep="first").reset_index(drop=True)

    dtypes = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    ratings = pd.read_csv(p_ratings, dtype=dtypes)
    if ratings_sample_fraction is not None and 0 < ratings_sample_fraction < 1.0:
        ratings = ratings.sample(frac=ratings_sample_fraction, random_state=42).reset_index(drop=True)

    return out, ratings


def dataset_status() -> dict:
    root = dataset_dir()
    ml = root / "ml-32m"
    files = {
        "MovieLens movies": ml / "movies.csv",
        "MovieLens links": ml / "links.csv",
        "MovieLens ratings": ml / "ratings.csv",
        "TMDB metadata (top_rated_movies)": root / "top_rated_movies.csv",
        "Wikipedia plots": root / "wiki_movie_plots_deduped.csv",
    }
    return {name: path.is_file() for name, path in files.items()}
