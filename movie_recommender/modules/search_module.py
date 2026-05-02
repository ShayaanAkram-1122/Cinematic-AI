"""BFS, DFS, and A* over a genre graph built from the filtered catalog."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from modules.heuristic_module import build_astar_norm_frame
from modules.utils import genre_match_score, map_ui_genre_to_ml, parse_movielens_genres


def _genre_sets(df: pd.DataFrame) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}
    for _, row in df.iterrows():
        mid = int(row["movieId"])
        gl = row["genres_list"] if isinstance(row["genres_list"], list) else parse_movielens_genres(str(row.get("genres", "")))
        out[mid] = set(gl)
    return out


def _build_adjacency(movie_ids: List[int], genre_sets: Dict[int, Set[str]]) -> Dict[int, List[int]]:
    max_neighbors = 80
    adj: Dict[int, List[int]] = {m: [] for m in movie_ids}
    for a in movie_ids:
        ga = genre_sets.get(a, set())
        scored: List[tuple[int, int]] = []
        for b in movie_ids:
            if a == b:
                continue
            inter = len(ga & genre_sets.get(b, set()))
            if inter > 0:
                scored.append((-inter, b))
        scored.sort()
        adj[a] = [b for _, b in scored[:max_neighbors]]
    return adj


def _pick_seed(df: pd.DataFrame, user_prefs: Dict[str, Any]) -> int:
    if df.empty:
        return -1
    j = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0).idxmax()
    return int(df.loc[j, "movieId"])


def bfs_search(df: pd.DataFrame, user_prefs: Dict[str, Any]) -> List[int]:
    if df.empty:
        return []
    movie_ids = df["movieId"].astype(int).tolist()
    gs = _genre_sets(df)
    adj = _build_adjacency(movie_ids, gs)
    seed = _pick_seed(df, user_prefs)
    if seed < 0 or seed not in adj:
        return movie_ids[:200]
    order: List[int] = []
    seen = {seed}
    q = deque([seed])
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    for m in movie_ids:
        if m not in seen:
            order.append(m)
    return order


def dfs_search(df: pd.DataFrame, user_prefs: Dict[str, Any]) -> List[int]:
    if df.empty:
        return []
    movie_ids = df["movieId"].astype(int).tolist()
    gs = _genre_sets(df)
    adj = _build_adjacency(movie_ids, gs)
    seed = _pick_seed(df, user_prefs)
    if seed < 0:
        return movie_ids[:200]
    order: List[int] = []
    seen: Set[int] = set()

    def dfs(u: int) -> None:
        if u in seen:
            return
        seen.add(u)
        order.append(u)
        for v in adj.get(u, []):
            dfs(v)

    dfs(seed)
    for m in movie_ids:
        if m not in seen:
            dfs(m)
    return order


def _h_movie(mid: int, id_to_row_idx: Dict[int, int], df: pd.DataFrame, norm: pd.DataFrame, user_prefs: Dict[str, Any]) -> float:
    pos = id_to_row_idx.get(mid)
    if pos is None:
        return 0.0
    row = df.loc[pos]
    nv = float(norm.loc[pos, "nv"])
    npv = float(norm.loc[pos, "np"])
    genres = row["genres_list"] if isinstance(row["genres_list"], list) else parse_movielens_genres(str(row.get("genres", "")))
    sel = {map_ui_genre_to_ml(g) for g in (user_prefs.get("genres") or [])}
    gm = genre_match_score(genres, sel) if sel else 1.0
    return float(0.4 * nv + 0.3 * npv + 0.3 * gm)


def astar_search(df: pd.DataFrame, user_prefs: Dict[str, Any], top_n: int = 50) -> Tuple[List[int], List[Tuple[int, float, float]]]:
    """Priority on f = g + h (maximize score → store -f in min-heap)."""
    if df.empty:
        return [], []
    df = df.reset_index(drop=True)
    norm = build_astar_norm_frame(df)
    movie_ids = df["movieId"].astype(int).tolist()
    id_to_row_idx = {int(r["movieId"]): i for i, r in df.iterrows()}
    gs = _genre_sets(df)
    adj = _build_adjacency(movie_ids, gs)
    seed = _pick_seed(df, user_prefs)
    if seed < 0 or seed not in adj:
        return movie_ids[:top_n], []

    pq: List[tuple[float, int, int]] = []
    h0 = _h_movie(seed, id_to_row_idx, df, norm, user_prefs)
    heapq.heappush(pq, (-(0.0 + h0), 0, seed))
    visited: Set[int] = set()
    log: List[Tuple[int, float, float]] = []
    out_order: List[int] = []

    cap = max(top_n * 6, 120)
    while pq and len(out_order) < cap:
        neg_f, g, mid = heapq.heappop(pq)
        if mid in visited:
            continue
        visited.add(mid)
        hf = _h_movie(mid, id_to_row_idx, df, norm, user_prefs)
        out_order.append(mid)
        log.append((mid, float(-neg_f), hf))
        for nb in adj.get(mid, []):
            if nb in visited:
                continue
            ng = g + 1
            nf = ng + _h_movie(nb, id_to_row_idx, df, norm, user_prefs)
            heapq.heappush(pq, (-nf, ng, nb))

    for m in movie_ids:
        if m not in visited:
            out_order.append(m)
    return out_order[: max(top_n, 50)], log
