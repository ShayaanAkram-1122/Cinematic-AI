"""Human-readable explanations for recommended movies."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from modules.utils import first_n_sentences, map_ui_genre_to_ml, parse_movielens_genres


def generate_explanation(
    movie_row: pd.Series,
    user_prefs: Dict[str, Any],
    scores_dict: Dict[str, float],
) -> Dict[str, Any]:
    genres = movie_row["genres_list"] if isinstance(movie_row["genres_list"], list) else parse_movielens_genres(str(movie_row.get("genres", "")))
    sel = [map_ui_genre_to_ml(g) for g in (user_prefs.get("genres") or [])]
    matched = [g for g in sel if g in set(genres)]
    why_parts = []
    if matched:
        why_parts.append("Matched your genres: " + ", ".join(matched))
    else:
        why_parts.append("Broad match to your filters and quality signals.")

    cid = scores_dict.get("cluster_id")
    if cid is not None and cid >= 0:
        why_parts.append(f"In taste cluster #{int(cid)} (K-means on movie features).")

    pr = scores_dict.get("predicted_rating")
    if pr is not None:
        why_parts.append(f"Predicted your rating: {pr:.1f} / 5.0")

    plot_src = str(movie_row.get("plot_summary") or movie_row.get("overview") or "")
    plot = first_n_sentences(plot_src, 3)

    dom = scores_dict.get("dominant_ai", "Heuristic")
    ai_reasoning = (
        f"Primary signal: {dom}. "
        f"Heuristic match {scores_dict.get('heuristic_score', 0):.2f}, "
        f"ANN norm {scores_dict.get('pred_norm', 0):.2f}, "
        f"cluster sim {scores_dict.get('cluster_similarity_score', 0):.2f}."
    )

    breakdown = {
        "heuristic": float(scores_dict.get("heuristic_score", 0)),
        "ann_norm": float(scores_dict.get("pred_norm", 0)),
        "cluster": float(scores_dict.get("cluster_similarity_score", 0)),
    }

    return {
        "why": " ".join(why_parts),
        "plot": plot,
        "ai_reasoning": ai_reasoning,
        "score_breakdown": breakdown,
    }
