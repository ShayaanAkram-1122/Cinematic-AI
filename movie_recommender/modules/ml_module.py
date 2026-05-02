"""K-means clustering and ANN (MLP) rating prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from modules.utils import extract_year_from_title, parse_movielens_genres

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ANN_PATH = MODELS_DIR / "ann_model.pkl"
ANN_META_PATH = MODELS_DIR / "ann_meta.json"


def _genre_lists_column(df: pd.DataFrame) -> list:
    gls = df["genres_list"].tolist()
    gss = df["genres"].tolist() if "genres" in df.columns else [""] * len(df)
    return [g if isinstance(g, list) else parse_movielens_genres(str(s)) for g, s in zip(gls, gss)]


def _genre_matrix(df: pd.DataFrame) -> np.ndarray:
    genres_list = _genre_lists_column(df)
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(genres_list), mlb


def cluster_movies(df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans, MultiLabelBinarizer]:
    if df.empty:
        return df.assign(cluster_id=-1), KMeans(n_clusters=1), MultiLabelBinarizer()
    Xg, mlb = _genre_matrix(df)
    year = pd.to_numeric(df["year"], errors="coerce").fillna(df["title"].map(extract_year_from_title)).fillna(2000).values.reshape(-1, 1)
    year = (year - year.mean()) / (year.std() + 1e-6)
    va = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0).values.reshape(-1, 1)
    pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).values.reshape(-1, 1)
    X = np.hstack([Xg, year, va, pop])
    km = KMeans(n_clusters=min(n_clusters, max(2, len(df) // 5)), random_state=random_state, n_init=5)
    labels = km.fit_predict(X)
    out = df.copy()
    out["cluster_id"] = labels
    return out, km, mlb


def find_similar_movies(movie_id: int, df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["movieId"] == int(movie_id)]
    if sub.empty or "cluster_id" not in df.columns:
        return pd.DataFrame()
    cid = int(sub["cluster_id"].iloc[0])
    return df[df["cluster_id"] == cid].sort_values("vote_average", ascending=False)


def _build_user_movie_sparse(ratings: pd.DataFrame, max_users: int = 4_000) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    ucounts = ratings["userId"].value_counts()
    top_users = ucounts.head(max_users).index.values
    sub = ratings[ratings["userId"].isin(top_users)]
    u_ids, u_inv = np.unique(sub["userId"].to_numpy(), return_inverse=True)
    m_ids, m_inv = np.unique(sub["movieId"].to_numpy(), return_inverse=True)
    data = sub["rating"].to_numpy(dtype=np.float32)
    mat = sparse.csr_matrix((data, (u_inv, m_inv)), shape=(len(u_ids), len(m_ids)))
    return mat, u_ids, m_ids


def cluster_users(
    ratings: pd.DataFrame,
    n_clusters: int = 10,
    max_users: int = 4_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, MiniBatchKMeans]:
    """MiniBatchKMeans on a CSR user–movie matrix (never densify the full matrix)."""
    if ratings.empty:
        return pd.DataFrame(columns=["userId", "user_cluster"]), MiniBatchKMeans(n_clusters=1)
    mat, u_ids, _ = _build_user_movie_sparse(ratings, max_users=max_users)
    if mat.shape[0] < 2:
        return pd.DataFrame({"userId": u_ids, "user_cluster": 0}), MiniBatchKMeans(n_clusters=1)
    n_clust = min(n_clusters, max(2, mat.shape[0] // 2))
    km = MiniBatchKMeans(
        n_clusters=n_clust,
        random_state=random_state,
        batch_size=min(2048, mat.shape[0]),
        n_init=3,
        max_iter=100,
    )
    km.fit(mat)
    labels = km.predict(mat)
    return pd.DataFrame({"userId": u_ids, "user_cluster": labels}), km


def train_or_load_ann(
    movies_df: pd.DataFrame,
    ratings: pd.DataFrame,
    max_train_rows: int = 120_000,
) -> Tuple[MLPRegressor, MultiLabelBinarizer, StandardScaler, Dict[str, Any]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if ANN_PATH.is_file() and ANN_META_PATH.is_file():
        bundle = joblib.load(ANN_PATH)
        meta = json.loads(ANN_META_PATH.read_text())
        return bundle["model"], bundle["mlb"], bundle["scaler"], meta

    movie_cols = ["movieId", "genres_list", "year", "popularity", "vote_average", "title"]
    if "genres" in movies_df.columns:
        movie_cols.append("genres")
    joined = ratings.merge(movies_df[movie_cols], on="movieId", how="inner")
    if len(joined) > max_train_rows:
        joined = joined.sample(n=max_train_rows, random_state=42)
    genres_list = _genre_lists_column(joined)
    mlb = MultiLabelBinarizer()
    Xg = mlb.fit_transform(genres_list)
    years = pd.to_numeric(joined["year"], errors="coerce").fillna(joined["title"].map(extract_year_from_title)).fillna(2000).values.reshape(-1, 1)
    pop = pd.to_numeric(joined["popularity"], errors="coerce").fillna(0).values.reshape(-1, 1)
    va = pd.to_numeric(joined["vote_average"], errors="coerce").fillna(0).values.reshape(-1, 1)
    Xnum = np.hstack([years / 2100.0, pop / 100.0, va / 10.0])
    X = np.hstack([Xg, Xnum])
    y = joined["rating"].to_numpy(dtype=np.float64)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    model = MLPRegressor(
        hidden_layer_sizes=(96, 48),
        max_iter=28,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=4,
    )
    model.fit(Xs, y)
    meta = {"train_rows": int(len(joined)), "loss": float(model.loss_) if hasattr(model, "loss_") else 0.0}
    joblib.dump({"model": model, "mlb": mlb, "scaler": scaler}, ANN_PATH)
    ANN_META_PATH.write_text(json.dumps(meta))
    return model, mlb, scaler, meta


def predict_ratings_batch(
    movie_ids: np.ndarray,
    model: MLPRegressor,
    mlb: MultiLabelBinarizer,
    scaler: StandardScaler,
    movies_df: pd.DataFrame,
) -> np.ndarray:
    """One batched `predict` call (much faster than thousands of single-row predicts)."""
    cols = ["movieId", "genres_list", "year", "popularity", "vote_average", "title"]
    if "genres" in movies_df.columns:
        cols.append("genres")
    feat = pd.DataFrame({"movieId": movie_ids}).merge(movies_df[cols], on="movieId", how="left")
    genres_list = _genre_lists_column(feat)
    Xg = mlb.transform(genres_list)
    years = (
        pd.to_numeric(feat["year"], errors="coerce")
        .fillna(feat["title"].map(extract_year_from_title))
        .fillna(2000)
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1)
    )
    pop = pd.to_numeric(feat["popularity"], errors="coerce").fillna(0).to_numpy(dtype=np.float64).reshape(-1, 1)
    va = pd.to_numeric(feat["vote_average"], errors="coerce").fillna(0).to_numpy(dtype=np.float64).reshape(-1, 1)
    Xnum = np.hstack([years / 2100.0, pop / 100.0, va / 10.0])
    X = np.hstack([Xg, Xnum])
    Xs = scaler.transform(X)
    pred = model.predict(Xs).astype(np.float64)
    return np.clip(pred, 0.5, 5.0)


def predict_rating(user_id: int, movie_id: int, model: MLPRegressor, mlb: MultiLabelBinarizer, scaler: StandardScaler, movies_df: pd.DataFrame) -> float:
    row = movies_df[movies_df["movieId"] == int(movie_id)]
    if row.empty:
        return 2.5
    return float(predict_ratings_batch(np.array([int(movie_id)]), model, mlb, scaler, movies_df)[0])


