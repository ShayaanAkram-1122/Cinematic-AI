# Cinematic AI

Hybrid movie recommendation system for **AI 2002**. Streamlit front end; backend combines constraint filtering, graph search (BFS / DFS / A*), heuristics, K-means clustering, an MLP-based rating predictor, and short recommendation explanations.

## What it does

- **Constraints** — Filter by genre, year, runtime, vote average, and popularity.
- **Search** — Traverse a genre-similarity graph with BFS, DFS, or A*.
- **Heuristics** — Rank candidates using normalized ratings, popularity, genre overlap, and revenue where available.
- **Clustering** — K-means on movies; MiniBatch K-means on sparse user–rating patterns.
- **Rating model** — `MLPRegressor` (scikit-learn) on engineered features; first train writes `movie_recommender/models/ann_model.pkl` and `ann_meta.json`.
- **Decision layer** — Fuses heuristic score, predicted rating, and cluster signals; optional second-user preferences.
- **UI** — Home, Preferences, pipeline visualizer, Recommendations, Analytics.

## Repository layout

The loader resolves `Dataset/` as a sibling of `movie_recommender/` (repository root):

```
<repo-root>/
├── Dataset/
│   ├── ml-32m/
│   │   ├── movies.csv
│   │   ├── links.csv
│   │   └── ratings.csv
│   ├── top_rated_movies.csv
│   └── wiki_movie_plots_deduped.csv
└── movie_recommender/
    ├── app.py
    ├── requirements.txt
    ├── models/
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

If you only have the app folder, add `Dataset/` at the same level or change the path logic in `movie_recommender/modules/data_loader.py`.

## Data sources

Place files so paths match the tree above.

| Data | Source |
|------|--------|
| MovieLens 32M | [GroupLens — MovieLens 32M](https://grouplens.org/datasets/movielens/32m/). ZIP: [ml-32m.zip](https://files.grouplens.org/datasets/movielens/ml-32m.zip). Extract into `Dataset/ml-32m/`. |
| TMDB-style CSV | Expected as `Dataset/top_rated_movies.csv` with columns usable by the loader (`id`, title, overview, dates, popularity, votes, etc.). Derive from [TMDB API](https://developer.themoviedb.org/docs/getting-started) or adapt a public dataset such as [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). |
| Wikipedia plots | Example: [Wikipedia Movie Plots (Kaggle)](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). Save as `Dataset/wiki_movie_plots_deduped.csv` or align columns with `data_loader.py`. |

`ratings.csv` and other large artifacts are listed in `.gitignore` and are not part of the repository. After cloning, add them locally under `Dataset/ml-32m/`.

## Requirements

- Python 3.10+ (virtual environment recommended).
- Sufficient RAM for full `ratings.csv`; use the in-app ratings sample control if needed.
- Disk space for uncompressed MovieLens files.

## Setup

From repository root:

```bash
git clone <repository-url>
cd <repository-root>

python3 -m venv movie_recommender/.venv
source movie_recommender/.venv/bin/activate   # Windows: movie_recommender\.venv\Scripts\activate
pip install -r movie_recommender/requirements.txt

streamlit run movie_recommender/app.py
```

Or from `movie_recommender/`:

```bash
cd movie_recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). Configure preferences, run the pipeline from the visualizer page, then inspect recommendations and analytics.

## Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Data | pandas, NumPy |
| ML | scikit-learn (KMeans, MiniBatchKMeans, MLPRegressor) |
| Sparse matrices | SciPy |
| Charts | Plotly |

`heapq` is from the standard library only. The neural component is sklearn’s MLP, not a separate deep-learning framework.

## Trained models

To force retraining, remove `movie_recommender/models/ann_model.pkl` and `ann_meta.json` (keep the directory if you rely on `.gitkeep`).

## Performance

- Reduce the ratings sample while developing.
- Movie clustering is cached for the session after the first run; user clustering uses sparse MiniBatch K-means.

## Colab

You can import `modules/` in a notebook with data on Drive or an uploaded `Dataset/`. The full Streamlit app in Colab would need port forwarding (e.g. ngrok). GPU is optional for the current pipeline.

## Course

AI 2002 — Hybrid AI Movie Recommendation & Decision System. Respect MovieLens, TMDB, and Wikipedia/Kaggle terms for redistributed data.
