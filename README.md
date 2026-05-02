# Cinematic AI — Hybrid Movie Recommender

**Course project (AI 2002)** — A Streamlit app that recommends movies by combining classical AI search, constraint satisfaction, clustering, a neural rating model, and short explainable summaries.

---

## Overview

**Cinematic AI** ingests MovieLens ratings, TMDB-style metadata, and Wikipedia plot text, merges them into one catalog, then runs a staged pipeline:

1. **CSP (constraint satisfaction)** — Hard filters on genres, year, runtime, vote average, and popularity.
2. **Graph search** — **BFS**, **DFS**, or **A\*** over a genre-similarity graph to order exploration.
3. **Heuristics** — Normalized scores (ratings, popularity, genre match, revenue) for ranking.
4. **K-means** — Movie clusters (full catalog) and user clusters (sparse rating patterns).
5. **ANN** — `MLPRegressor` predicts a 0–5 rating from genre encoding + year + popularity + vote average (trained once, saved under `models/`).
6. **Decision fusion** — Blends heuristic, predicted rating, and cluster similarity; optional **multi-user** compromise layer.
7. **Explainability** — “Why recommended,” plot snippets, and score breakdowns in the UI.

The interface is a **cinematic dark theme** with five sections: Home, Preferences, AI Visualizer, Recommendations, and Analytics.

---

## Repository layout

The data loader expects this structure **next to** the `movie_recommender` folder (i.e. parent of `movie_recommender` on disk):

```
<your-repo-root>/
├── Dataset/
│   ├── ml-32m/
│   │   ├── movies.csv
│   │   ├── links.csv
│   │   └── ratings.csv
│   ├── top_rated_movies.csv      # TMDB-style; joined via links.tmdbId
│   └── wiki_movie_plots_deduped.csv
└── movie_recommender/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── models/                     # created on first ANN train (.pkl + meta)
    └── modules/
        ├── data_loader.py
        ├── csp_module.py
        ├── search_module.py
        ├── heuristic_module.py
        ├── ml_module.py
        ├── decision_module.py
        ├── explainability.py
        └── utils.py
```

If your clone only contains `movie_recommender/`, add a `Dataset/` sibling or edit `dataset_dir()` in `modules/data_loader.py`.

---

## Requirements

- **Python 3.10+** (3.11–3.14 tested in development; use a venv).
- **RAM**: MovieLens `ratings.csv` is large; use the in-app **ratings sample** slider (e.g. 1–10%) if loading is slow or memory is tight.
- **Disk**: full ml-32m ratings are multi‑GB uncompressed.

---

## Quick start (GitHub clone)

```bash
git clone <your-repo-url>
cd <your-repo-root>

python3 -m venv movie_recommender/.venv
source movie_recommender/.venv/bin/activate          # Windows: movie_recommender\.venv\Scripts\activate
pip install -r movie_recommender/requirements.txt

streamlit run movie_recommender/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

**Workflow:** **Home** → **Preferences** (set user ID, genres, sliders) → **FIND MY MOVIES** → **AI Visualizer** (runs the pipeline) → **Recommendations** / **Analytics**.

---

## Install (from inside `movie_recommender/`)

```bash
cd movie_recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the same app from either location; the working directory only affects relative paths if you change code—**data paths are resolved from the folder that contains `Dataset/`** as described above.

---

## Tech stack

| Area | Choice |
|------|--------|
| UI | [Streamlit](https://streamlit.io/) |
| Data | pandas, numpy |
| ML | scikit-learn (KMeans, MiniBatchKMeans, MLPRegressor, TF-IDF-style genre encoding) |
| Sparse users | scipy.sparse |
| Viz | Plotly |

**Note:** There is no `heapq` pip package (it is in the Python standard library). The “ANN” is implemented with **scikit-learn `MLPRegressor`** (not TensorFlow) for simpler installs and CPU-friendly training.

---

## Model cache

After the first successful training, weights are saved as:

- `movie_recommender/models/ann_model.pkl`
- `movie_recommender/models/ann_meta.json`

Delete both files to force a full retrain (e.g. after changing feature engineering).

---

## Performance tips

- Lower **Ratings load sample** on Preferences while iterating.
- First run in a session pays for **movie K-means** on the full catalog; later runs reuse cached clusters until data is reloaded.
- User clustering uses **MiniBatchKMeans on a sparse matrix** (no full dense user×movie matrix).

---

## Google Colab

The **logic** in `modules/` can be imported from a notebook if you upload or mount `Dataset/`. Running the **full Streamlit UI** in Colab needs tunneling (e.g. ngrok) and is optional. GPU is not required for the current sklearn pipeline.

---

## Authors / course

Submitted as **AI 2002 — Hybrid AI Movie Recommendation & Decision System**. Adapt dataset paths and citations to match your institution’s policy (MovieLens / TMDB / Wikipedia terms of use).
