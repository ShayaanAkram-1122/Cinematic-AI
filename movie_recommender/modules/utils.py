"""Shared helpers for the movie recommender."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

import numpy as np
import pandas as pd


def extract_year_from_title(title: str) -> float:
    if not isinstance(title, str) or not title:
        return np.nan
    m = re.search(r"\((\d{4})\)\s*$", title.strip())
    if m:
        return float(m.group(1))
    return np.nan


def strip_year_from_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title.strip()).strip()


def normalize_title(title: str) -> str:
    t = strip_year_from_title(title) if title else ""
    return t.lower().strip()


def parse_movielens_genres(genres_str: str) -> List[str]:
    if not isinstance(genres_str, str) or not genres_str:
        return []
    return [g.strip() for g in genres_str.split("|") if g and g != "(no genres listed)"]


def genres_to_pipe_list(genres: Iterable[str]) -> str:
    return "|".join(sorted({g for g in genres if g}))


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def first_n_sentences(text: str, n: int = 3) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in parts if s]
    return " ".join(sentences[:n]).strip()


USER_GENRE_OPTIONS = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Western",
]


def map_ui_genre_to_ml(ui_genre: str) -> str:
    """MovieLens uses Sci-Fi; UI matches."""
    return ui_genre


def genre_match_score(movie_genres: List[str], selected: Set[str]) -> float:
    if not selected:
        return 1.0
    movie_set = set(movie_genres)
    if not movie_set:
        return 0.0
    inter = movie_set & selected
    return len(inter) / max(len(selected), 1)
