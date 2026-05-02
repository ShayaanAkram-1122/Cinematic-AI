"""
Cinematic AI — Hybrid Movie Recommendation & Decision System (AI 2002).
Run from repo root:  streamlit run movie_recommender/app.py
Or from this folder: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import html
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from modules.csp_module import CSPFilter, constraint_filter_counts
from modules.data_loader import dataset_status, load_merged_movies_and_ratings
from modules.decision_module import build_recommendations, minimax_recommendations
from modules.explainability import generate_explanation
from modules.ml_module import cluster_movies, cluster_users, train_or_load_ann
from modules.search_module import astar_search, bfs_search, dfs_search
from modules.utils import USER_GENRE_OPTIONS, first_n_sentences

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-card: #12121a;
    --bg-elevated: #1a1a2e;
    --accent-primary: #e50914;
    --accent-secondary: #f5a623;
    --accent-glow: rgba(229, 9, 20, 0.3);
    --text-primary: #ffffff;
    --text-secondary: #a0a0b0;
    --border: rgba(255,255,255,0.08);
    --font-display: 'Bebas Neue', cursive;
    --font-body: 'DM Sans', sans-serif;
}

body, .stApp { background-color: var(--bg-primary); color: var(--text-primary); font-family: var(--font-body); }

.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent-primary);
}
.movie-card:hover { border-color: var(--accent-primary); box-shadow: 0 0 20px var(--accent-glow); }

.rank-badge {
    background: var(--accent-primary);
    color: white;
    font-family: var(--font-display);
    font-size: 1.4rem;
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
}

.score-bar-container { background: rgba(255,255,255,0.05); border-radius: 4px; height: 6px; margin: 4px 0; }
.score-bar { background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)); border-radius: 4px; height: 6px; }

.ai-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}
.badge-astar { background: rgba(229,9,20,0.2); border: 1px solid var(--accent-primary); color: var(--accent-primary); }
.badge-kmeans { background: rgba(245,166,35,0.2); border: 1px solid var(--accent-secondary); color: var(--accent-secondary); }
.badge-ann { background: rgba(100,200,100,0.2); border: 1px solid #64c864; color: #64c864; }
.badge-csp { background: rgba(100,150,255,0.2); border: 1px solid #6496ff; color: #6496ff; }

.section-title { font-family: var(--font-display); font-size: 2rem; letter-spacing: 2px; color: var(--text-primary); }
.plot-box { background: var(--bg-elevated); border-left: 3px solid var(--accent-secondary); padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6; margin: 8px 0; }

/* App shell */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    color: rgba(255,255,255,0.5) !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Bebas Neue', cursive !important;
    font-size: 22px !important;
    letter-spacing: 3px;
    color: white !important;
}
.stApp { background-color: #080810 !important; }
</style>
"""

HOME_SECTION_CSS = """
<style>
/* Hero + stats + pipeline + dataset (home) */
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: clamp(64px, 10vw, 120px);
    line-height: 0.9;
    color: #ffffff;
    letter-spacing: 4px;
    margin: 0;
}
.hero-title span {
    color: #e50914;
}
.hero-tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin: 16px 0 40px;
    font-weight: 300;
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
    margin: 48px 0;
}
.stat-cell {
    background: #0d0d18;
    padding: 28px 24px;
    position: relative;
}
.stat-cell::after {
    content: '';
    position: absolute;
    bottom: 0; left: 24px; right: 24px;
    height: 2px;
    background: #e50914;
    transform: scaleX(0);
    transition: transform 0.3s ease;
    border-radius: 2px;
}
.stat-cell:hover::after { transform: scaleX(1); }
.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.3);
    font-weight: 500;
    margin-bottom: 10px;
}
.stat-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 48px;
    color: #ffffff;
    line-height: 1;
    letter-spacing: 2px;
}
.stat-value span { color: #e50914; }
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 8px 0 48px;
}
.cta-wrap { margin: 0 0 56px; }
.section-heading {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: rgba(255,255,255,0.3);
    font-weight: 500;
    margin-bottom: 20px;
}
.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 56px;
}
.pipe-card {
    background: #0d0d18;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 24px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.pipe-card:hover { border-color: rgba(229,9,20,0.4); }
.pipe-card::before {
    content: attr(data-num);
    font-family: 'Bebas Neue', cursive;
    font-size: 80px;
    color: rgba(255,255,255,0.03);
    position: absolute;
    right: 10px;
    top: -10px;
    line-height: 1;
}
.pipe-num {
    font-family: 'Bebas Neue', cursive;
    font-size: 13px;
    color: #e50914;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.pipe-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: #ffffff;
    margin-bottom: 8px;
    line-height: 1.3;
}
.pipe-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: rgba(255,255,255,0.35);
    line-height: 1.6;
    font-weight: 300;
}
.dataset-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.ds-card {
    background: #0d0d18;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.ds-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px rgba(34,197,94,0.6);
    flex-shrink: 0;
}
.ds-dot.ds-bad {
    background: #ef4444;
    box-shadow: 0 0 8px rgba(239,68,68,0.5);
}
.ds-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: rgba(255,255,255,0.7);
    line-height: 1.4;
}
.ds-tag {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #22c55e;
    font-weight: 500;
}
.ds-tag.ds-bad { color: #ef4444; }
@media (max-width: 1100px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .pipeline-grid { grid-template-columns: repeat(2, 1fr); }
    .dataset-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
    .stat-grid, .pipeline-grid, .dataset-grid { grid-template-columns: 1fr; }
}
</style>
"""


def init_session() -> None:
    defaults = {
        "reco_df": None,
        "reco_meta": None,
        "feedback": {},
        "prefs": {},
        "pipeline_log": {},
        "movies_df": None,
        "ratings_df": None,
        "pending_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def popularity_from_tier(tier: str, pop_series: pd.Series) -> float:
    s = pd.to_numeric(pop_series, errors="coerce").fillna(0)
    q33, q66 = float(s.quantile(0.33)), float(s.quantile(0.66))
    if tier == "Low":
        return q33
    if tier == "High":
        return q66
    return q33 + (q66 - q33) * 0.25


def run_pipeline(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    prefs: Dict[str, Any],
    prefs_b: Optional[Dict[str, Any]],
    search_algo: str,
    multi_user: bool,
) -> None:
    log: Dict[str, Any] = {}
    filtered = CSPFilter.apply_all_constraints(movies_df, prefs)
    log["csp_count"] = len(filtered)
    if filtered.empty:
        st.session_state["reco_df"] = pd.DataFrame()
        st.session_state["reco_meta"] = {"error": "No movies matched your constraints. Relax filters."}
        st.session_state["pipeline_log"] = log
        return

    if search_algo == "BFS":
        ordered = bfs_search(filtered, prefs)
    elif search_algo == "DFS":
        ordered = dfs_search(filtered, prefs)
    else:
        ordered, astar_log = astar_search(filtered, prefs, top_n=prefs.get("top_n", 15))
        log["astar_log"] = astar_log[:80]
    log["search_algo"] = search_algo
    log["search_nodes"] = len(ordered)

    # K-means on ~87k movies is expensive; reuse for the same loaded `movies_df` object in this session.
    mid = id(movies_df)
    if st.session_state.get("_cluster_movies_id") != mid:
        st.session_state["_cluster_movies_id"] = mid
        st.session_state["_cluster_movies_df"], _, _ = cluster_movies(movies_df, n_clusters=prefs.get("k_clusters", 10))
    movies_clustered = st.session_state["_cluster_movies_df"]

    rid = id(ratings_df)
    if st.session_state.get("_cluster_users_id") != rid:
        st.session_state["_cluster_users_id"] = rid
        st.session_state["_cluster_users_table"], _ = cluster_users(ratings_df, n_clusters=10, max_users=4_000)
    user_clusters = st.session_state["_cluster_users_table"]
    uid = int(prefs["user_id"])
    urow = user_clusters[user_clusters["userId"] == uid]
    user_cluster_row = urow.iloc[0] if not urow.empty else None
    log["user_cluster"] = int(user_cluster_row["user_cluster"]) if user_cluster_row is not None else None

    model, mlb, scaler, ann_meta = train_or_load_ann(movies_df, ratings_df)
    log["ann_meta"] = ann_meta

    if ratings_df.empty or uid not in set(ratings_df["userId"].unique()):
        log["user_missing"] = True
    else:
        log["user_missing"] = False

    ranked = build_recommendations(
        filtered,
        ordered,
        prefs,
        movies_clustered,
        user_cluster_row,
        model,
        mlb,
        scaler,
        uid,
    )
    top_n = int(prefs.get("top_n", 10))
    out = ranked.head(top_n).copy()

    if multi_user and prefs_b is not None and bool(prefs.get("multi_user")):
        compromise = minimax_recommendations(filtered, prefs, prefs_b)
        log["multiuser_head"] = compromise.head(8)
        # Re-rank top list: boost if in compromise head
        head_ids = set(compromise["movieId"].astype(int).head(8).tolist())
        out["final_score"] = out["final_score"] + out["movieId"].apply(lambda m: 0.04 if int(m) in head_ids else 0.0)
        out = out.sort_values("final_score", ascending=False).head(top_n)

    st.session_state["reco_df"] = out
    st.session_state["reco_meta"] = {
        "prefs": prefs,
        "prefs_b": prefs_b,
        "constraint_table": constraint_filter_counts(movies_df, prefs),
        "movies_clustered_sample": movies_clustered.sample(min(800, len(movies_clustered))),  # for PCA plot
        "full_clustered": None,
        "user_cluster_row": user_cluster_row,
        "filtered_count": len(filtered),
    }
    st.session_state["pipeline_log"] = log


def _dataset_status_cards_html() -> str:
    parts: List[str] = []
    for name, ok in dataset_status().items():
        dot_cls = "ds-dot" if ok else "ds-dot ds-bad"
        tag = "✓ Loaded" if ok else "✗ Missing"
        tag_cls = "ds-tag" if ok else "ds-tag ds-bad"
        safe = html.escape(str(name))
        parts.append(
            f'<div class="ds-card"><div class="{dot_cls}"></div>'
            f'<div class="ds-name">{safe}</div><div class="{tag_cls}">{tag}</div></div>'
        )
    return "\n".join(parts)


def render_home(movies_df: Optional[pd.DataFrame], ratings_df: Optional[pd.DataFrame]) -> None:
    st.markdown(THEME, unsafe_allow_html=True)
    st.markdown(HOME_SECTION_CSS, unsafe_allow_html=True)

    st.markdown(
        """
    <p class="hero-title">CINEMATIC<span> AI</span></p>
    <p class="hero-tagline">A* Search &nbsp;·&nbsp; CSP &nbsp;·&nbsp; K-Means &nbsp;·&nbsp; Neural Networks &nbsp;·&nbsp; Explainable AI</p>
    """,
        unsafe_allow_html=True,
    )

    if movies_df is not None and ratings_df is not None:
        n_movies = f"{len(movies_df):,}"
        n_ratings = f"{len(ratings_df):,}"
        n_users = f"{ratings_df['userId'].nunique():,}"
        n_genres = f"{movies_df['genres'].astype(str).str.split('|').explode().nunique():,}"
    else:
        n_movies = n_ratings = n_users = n_genres = "—"

    st.markdown(
        f"""
    <div class="stat-grid">
        <div class="stat-cell">
            <div class="stat-label">Movies in database</div>
            <div class="stat-value">{n_movies}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Ratings loaded</div>
            <div class="stat-value">{n_ratings}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Unique users</div>
            <div class="stat-value">{n_users}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-label">Genre tags</div>
            <div class="stat-value"><span>{n_genres}</span></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cta-wrap">', unsafe_allow_html=True)
    if st.button("GET STARTED — SET PREFERENCES", key="cta_home", type="primary"):
        st.session_state["nav"] = "Preferences"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div class="pipeline-grid">
        <div class="pipe-card" data-num="01">
            <div class="pipe-num">STEP 01</div>
            <div class="pipe-title">Constraint Satisfaction</div>
            <div class="pipe-desc">CSP filters the catalog down to valid candidates using your genre, year, runtime and rating constraints.</div>
        </div>
        <div class="pipe-card" data-num="02">
            <div class="pipe-num">STEP 02</div>
            <div class="pipe-title">Graph Search Exploration</div>
            <div class="pipe-desc">BFS, DFS and A* traverse a genre graph to discover diverse, ranked movie candidates.</div>
        </div>
        <div class="pipe-card" data-num="03">
            <div class="pipe-num">STEP 03</div>
            <div class="pipe-title">ML Taste Modelling</div>
            <div class="pipe-desc">K-Means clusters movies and users. ANN predicts your rating for each shortlisted title.</div>
        </div>
        <div class="pipe-card" data-num="04">
            <div class="pipe-num">STEP 04</div>
            <div class="pipe-title">Explainable Decision</div>
            <div class="pipe-desc">Scores are fused and each recommendation is explained with plot summaries and AI reasoning.</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Dataset status</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dataset-grid">{_dataset_status_cards_html()}</div>', unsafe_allow_html=True)


def page_preferences() -> Dict[str, Any]:
    st.markdown(THEME, unsafe_allow_html=True)
    st.header("User preferences")
    uid = st.number_input("MovieLens user ID", min_value=1, max_value=200948, value=1, step=1)
    genres = st.multiselect("Genres", USER_GENRE_OPTIONS, default=["Drama", "Thriller"])
    y1, y2 = st.slider("Release year range", 1950, 2024, (1990, 2020))
    r1, r2 = st.slider("Runtime (minutes)", 60, 240, (90, 200))
    st.caption(f"Runtime span: {r1 // 60}h{r1 % 60:02d}m — {r2 // 60}h{r2 % 60:02d}m")
    min_rating = st.slider("Minimum TMDB-style vote average", 0.0, 10.0, 5.5, 0.5)
    pop_tier = st.select_slider("Popularity threshold", ["Low", "Medium", "High"], value="Medium")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    search_algo = st.radio("Search algorithm", ("A*", "BFS", "DFS"), horizontal=True)
    multi = st.checkbox("Multi-user mode (compromise ranking)", value=False)
    prefs_b = None
    if multi:
        st.subheader("Second viewer")
        uid_b = st.number_input("User B ID", 1, 200948, 2)
        gb = st.multiselect("Genres (user B)", USER_GENRE_OPTIONS, default=["Comedy", "Romance"])
        yb1, yb2 = st.slider("Year range (user B)", 1950, 2024, (1980, 2015))
        rb1, rb2 = st.slider("Runtime (user B)", 60, 240, (85, 190))
        mnb = st.slider("Min rating (user B)", 0.0, 10.0, 5.0, 0.5)
        prefs_b = {
            "user_id": int(uid_b),
            "genres": gb,
            "year_min": float(yb1),
            "year_max": float(yb2),
            "runtime_min": float(rb1),
            "runtime_max": float(rb2),
            "min_rating": float(mnb),
            "min_popularity": 0.0,
        }
    frac = st.slider("Ratings load sample (lower = less RAM)", 0.01, 1.0, 0.05, 0.01)
    if st.button("FIND MY MOVIES", type="primary"):
        st.session_state["ratings_sample"] = float(frac)
        st.session_state["movies_df"] = None
        st.session_state["ratings_df"] = None
        st.session_state["nav"] = "AI Visualizer"
        st.session_state["pending_run"] = True
        st.rerun()
    return {
        "user_id": int(uid),
        "genres": genres,
        "year_min": float(y1),
        "year_max": float(y2),
        "runtime_min": float(r1),
        "runtime_max": float(r2),
        "min_rating": float(min_rating),
        "min_popularity": 0.0,
        "top_n": int(top_n),
        "search_algo": search_algo,
        "multi_user": multi,
        "prefs_b": prefs_b,
        "pop_tier": pop_tier,
    }


def page_visualizer(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    st.markdown(THEME, unsafe_allow_html=True)
    st.header("AI processing visualizer")
    prefs = dict(st.session_state.get("prefs") or {})
    if not prefs:
        st.warning("Set preferences on the Preferences page first.")
        return
    pop_min = popularity_from_tier(prefs.get("pop_tier", "Medium"), movies_df["popularity"])
    prefs["min_popularity"] = pop_min
    if prefs.get("prefs_b") is not None:
        prefs["prefs_b"] = {**prefs["prefs_b"], "min_popularity": pop_min}

    if st.session_state.get("pending_run"):
        steps = [
            "CSP Filtering",
            f"{prefs.get('search_algo', 'A*')} Search",
            "Heuristic Ranking",
            "K-Means Clustering",
            "ANN Rating Prediction",
            "Decision Fusion",
        ]
        prog = st.progress(0)
        status = st.status("Running hybrid pipeline…", expanded=True)
        try:
            with status:
                for i, label in enumerate(steps):
                    time.sleep(0.12)
                    prog.progress((i + 1) / len(steps))
                    st.write(f"Step {i+1}: {label} — running…")
                run_pipeline(
                    movies_df,
                    ratings_df,
                    prefs,
                    prefs.get("prefs_b") if prefs.get("multi_user") else None,
                    str(prefs.get("search_algo", "A*")),
                    bool(prefs.get("multi_user")),
                )
                status.update(label="Pipeline complete", state="complete")
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline failed: {e}")
            st.session_state["pending_run"] = False
            prog.progress(1.0)
            return
        prog.progress(1.0)
        st.session_state["pending_run"] = False

    log = st.session_state.get("pipeline_log") or {}
    st.subheader("Step summary")
    st.write(f"CSP filtered candidates: **{log.get('csp_count', '—')}**")
    st.write(f"Search nodes ordered: **{log.get('search_nodes', '—')}** ({log.get('search_algo', '')})")
    if log.get("astar_log"):
        with st.expander("A* exploration (sample)"):
            st.write(pd.DataFrame(log["astar_log"], columns=["movieId", "f", "h"]).head(40))
    meta = st.session_state.get("reco_meta") or {}
    mc = meta.get("movies_clustered_sample")
    if isinstance(mc, pd.DataFrame) and not mc.empty and "cluster_id" in mc.columns:
        st.subheader("K-means (PCA-2D)")
        try:
            Xg = mc["genres"].astype(str).str.get_dummies(sep="|")
            X = pd.concat([Xg.reset_index(drop=True), mc[["vote_average", "popularity"]].reset_index(drop=True)], axis=1)
            X = X.fillna(0)
            pca = PCA(n_components=2, random_state=42)
            xy = pca.fit_transform(X.values)
            vis = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cluster": mc["cluster_id"].astype(str), "title": mc["title"].values})
            fig = px.scatter(vis, x="x", y="y", color="cluster", hover_data=["title"], title="Movie clusters (PCA)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"PCA plot skipped: {e}")

    reco = st.session_state.get("reco_df")
    if isinstance(reco, pd.DataFrame) and not reco.empty and "predicted_rating" in reco.columns:
        st.subheader("ANN predicted ratings (this result set)")
        fig2 = px.histogram(reco, x="predicted_rating", nbins=20, title="Predicted rating distribution")
        st.plotly_chart(fig2, use_container_width=True)


def score_bar_html(label: str, pct: float) -> str:
    w = max(0, min(100, int(pct * 100)))
    return f"""<div><span style="color:#a0a0b0;font-size:0.85rem;">{label}</span>
    <div class="score-bar-container"><div class="score-bar" style="width:{w}%"></div></div></div>"""


def page_recommendations() -> None:
    st.markdown(THEME, unsafe_allow_html=True)
    st.header("Recommendations")
    reco = st.session_state.get("reco_df")
    meta = st.session_state.get("reco_meta") or {}
    if isinstance(reco, pd.DataFrame) and reco.empty:
        st.error(meta.get("error", "No recommendations yet. Run the pipeline from Preferences → AI Visualizer."))
        return
    if not isinstance(reco, pd.DataFrame) or reco.empty:
        st.info("Run the pipeline first.")
        return

    sort_mode = st.selectbox("Sort by", ["Final Score", "Predicted Rating", "Popularity", "Year"])
    view = reco.copy()
    if sort_mode == "Predicted Rating":
        view = view.sort_values("predicted_rating", ascending=False)
    elif sort_mode == "Popularity":
        view = view.sort_values("popularity", ascending=False)
    elif sort_mode == "Year":
        view = view.sort_values("year", ascending=False)

    prefs = meta.get("prefs") or st.session_state.get("prefs") or {}
    movies_full = st.session_state.get("movies_df")
    for rank, (_, row) in enumerate(view.iterrows(), start=1):
        title = str(row.get("title", "Unknown"))
        year = int(row["year"]) if pd.notna(row.get("year")) else "—"
        genres = str(row.get("genres", "")).replace("|", " · ")
        pr = float(row.get("predicted_rating", 0))
        fs = float(row.get("final_score", 0))
        dom = str(row.get("dominant_ai", ""))
        cid = int(row.get("cluster_id", -1))
        expl = generate_explanation(
            row,
            prefs,
            {
                "heuristic_score": float(row.get("heuristic_score", 0)),
                "pred_norm": float(row.get("pred_norm", 0)),
                "cluster_similarity_score": float(row.get("cluster_similarity_score", 0)),
                "predicted_rating": pr,
                "cluster_id": cid,
                "dominant_ai": dom,
            },
        )
        card = f"""
<div class="movie-card">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span class="rank-badge">#{rank}</span>
    <strong>{title} ({year})</strong>
    <span style="color:#f5a623;">Score {fs:.2f}</span>
  </div>
  <p style="color:#a0a0b0;">{genres}</p>
  <p>⏱ {int(row.get('runtime',0))} min | ⭐ {row.get('vote_average',0):.1f} | 🔥 {row.get('popularity',0):.1f}</p>
  <div>
    <span class="ai-badge badge-astar">A* / Search</span>
    <span class="ai-badge badge-kmeans">K-Means #{cid}</span>
    <span class="ai-badge badge-ann">ANN {pr:.1f}★</span>
    <span class="ai-badge badge-csp">CSP ✓</span>
  </div>
  <div class="plot-box"><strong>Plot</strong><br/>{expl["plot"]}</div>
  <p><strong>Why recommended</strong><br/>{expl["why"]}</p>
  <p style="font-size:0.9rem;color:#a0a0b0;">{expl["ai_reasoning"]}</p>
  {score_bar_html("Heuristic", expl["score_breakdown"]["heuristic"])}
  {score_bar_html("ANN (norm)", expl["score_breakdown"]["ann_norm"])}
  {score_bar_html("Cluster sim", expl["score_breakdown"]["cluster"])}
</div>
"""
        st.markdown(card, unsafe_allow_html=True)
        with st.expander("Full plot / overview"):
            st.write(first_n_sentences(str(row.get("plot_summary") or row.get("overview") or ""), 25))
        fb = st.session_state["feedback"].get(int(row["movieId"]), None)
        c1, c2 = st.columns(2)
        if c1.button("Good rec", key=f"up_{row['movieId']}"):
            st.session_state["feedback"][int(row["movieId"])] = "up"
            st.rerun()
        if c2.button("Bad rec", key=f"down_{row['movieId']}"):
            st.session_state["feedback"][int(row["movieId"])] = "down"
            st.rerun()
        if fb:
            st.caption(f"Your feedback: {fb}")

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Download recommendations CSV", csv, file_name="recommendations.csv", mime="text/csv")


def page_analytics() -> None:
    st.markdown(THEME, unsafe_allow_html=True)
    st.header("Analytics")
    reco = st.session_state.get("reco_df")
    meta = st.session_state.get("reco_meta") or {}
    if not isinstance(reco, pd.DataFrame) or reco.empty:
        st.info("No recommendation batch to analyze yet.")
        return
    genres = reco["genres"].astype(str).str.split("|").explode().value_counts().head(12)
    fig = px.pie(values=genres.values, names=genres.index, title="Genre mix (recommended)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.histogram(reco, x="final_score", nbins=15, title="Final score distribution")
    st.plotly_chart(fig2, use_container_width=True)
    reco = reco.copy()
    reco["decade"] = (pd.to_numeric(reco["year"], errors="coerce").fillna(2000) // 10 * 10).astype(int).astype(str) + "s"
    fig3 = px.bar(reco["decade"].value_counts().reset_index(), x="decade", y="count", title="Decades")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("AI technique contribution (mean normalized signals)")
    tech = pd.DataFrame(
        {
            "technique": ["Heuristic / A*", "ANN", "K-Means"],
            "weight": [
                float(reco["heuristic_score"].mean()),
                float(reco["pred_norm"].mean()),
                float(reco["cluster_similarity_score"].mean()),
            ],
        }
    )
    st.plotly_chart(px.bar(tech, x="technique", y="weight", title="Average component strength"), use_container_width=True)

    ct = meta.get("constraint_table")
    if isinstance(ct, pd.DataFrame):
        st.subheader("CSP constraint summary")
        st.dataframe(ct, use_container_width=True)

    ucr = meta.get("user_cluster_row")
    if ucr is not None and isinstance(ucr, pd.Series):
        st.subheader("User cluster")
        st.write(f"User maps to cluster **{int(ucr.get('user_cluster', -1))}** (K-means on sparse rating patterns, sampled users).")

    prefs = meta.get("prefs") or {}
    uid = int(prefs.get("user_id", 0))
    ratings_df = st.session_state.get("ratings_df")
    if isinstance(ratings_df, pd.DataFrame) and uid and uid in ratings_df["userId"].values:
        hist = ratings_df[ratings_df["userId"] == uid].merge(reco[["movieId", "predicted_rating"]], on="movieId", how="inner")
        if not hist.empty:
            hist["actual"] = hist["rating"]
            fig4 = px.scatter(hist, x="actual", y="predicted_rating", hover_data=["movieId"], title="Actual vs predicted (overlap)")
            st.plotly_chart(fig4, use_container_width=True)
            mae = float((hist["actual"] - hist["predicted_rating"]).abs().mean())
            st.caption(f"Mean absolute error on overlap: {mae:.3f}")


def main() -> None:
    st.set_page_config(page_title="Cinematic AI", layout="wide", initial_sidebar_state="expanded")
    init_session()
    st.sidebar.markdown("### Navigation")
    nav_options = ["Home", "Preferences", "AI Visualizer", "Recommendations", "Analytics"]
    want = st.session_state.get("nav", "Home")
    nav_idx = nav_options.index(want) if want in nav_options else 0
    page = st.sidebar.radio("Page", nav_options, index=nav_idx)
    st.session_state["nav"] = page

    movies_df = st.session_state.get("movies_df")
    ratings_df = st.session_state.get("ratings_df")
    if movies_df is None:
        try:
            frac = float(st.session_state.get("ratings_sample", 0.05))
            m, r = load_merged_movies_and_ratings(
                ratings_sample_fraction=frac if frac < 1.0 else None,
            )
            st.session_state["movies_df"] = m
            st.session_state["ratings_df"] = r
            movies_df, ratings_df = m, r
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Data load error: {e}")
            return

    if page == "Home":
        render_home(movies_df, ratings_df)
    elif page == "Preferences":
        p = page_preferences()
        pop_min = popularity_from_tier(p.get("pop_tier", "Medium"), movies_df["popularity"])
        pb = p.get("prefs_b")
        if pb is not None:
            pb = {**pb, "min_popularity": pop_min}
        st.session_state["prefs"] = {
            "user_id": p["user_id"],
            "genres": p["genres"],
            "year_min": p["year_min"],
            "year_max": p["year_max"],
            "runtime_min": p["runtime_min"],
            "runtime_max": p["runtime_max"],
            "min_rating": p["min_rating"],
            "min_popularity": pop_min,
            "top_n": p["top_n"],
            "search_algo": p["search_algo"],
            "multi_user": p["multi_user"],
            "prefs_b": pb,
            "pop_tier": p.get("pop_tier", "Medium"),
        }
    elif page == "AI Visualizer":
        page_visualizer(movies_df, ratings_df)
    elif page == "Recommendations":
        page_recommendations()
    else:
        page_analytics()


if __name__ == "__main__":
    main()
