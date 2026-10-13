# Hybrid Bollywood Movie Recommendation System

SPPU BE Computer Engineering — Machine Learning mini-project.

The project demonstrates three recommendation approaches and a weighted hybrid:
1. **Content-based filtering:** TF-IDF over movie title, genres, cast, director, and synopsis; cosine similarity ranks similar movies.
2. **Item-based collaborative filtering:** movie similarity learned from user-rating patterns.
3. **SVD matrix factorisation:** truncated SVD learns latent factors from the user–movie ratings matrix.
4. **Hybrid recommender:** normalises and combines the three signals into a final ranking.

> Important: collaborative filtering needs actual user–movie ratings. Movie metadata and aggregate IMDb ratings alone are not user-level rating interactions.

## Dataset links

**Recommended aligned dataset source (Bollywood metadata + collaborative rating data):**
- TIMDB GitHub repository: https://github.com/pncnmnp/TIMDB
- The repository describes `1950-2019/bollywood_full.csv` as the merged Bollywood metadata file. It also documents collaborative files such as `ratings.csv`, `links.csv`, and `titles.csv`.
- Download/clone the repository and copy these files into this project's `data/` directory. The loader searches subfolders recursively, so keeping the repository's subfolder structure under `data/` is fine.

**Alternative content-only dataset (if you want a smaller, simpler starting point):**
- Bollywood Movie Dataset (about 2,200 films; title, year, genre, overview, director, cast): https://github.com/devensinghbhagtani/Bollywood-Movie-Dataset
- This alternative is useful for content-based recommendations, but it does not by itself provide user–movie ratings for collaborative filtering.

**Alternative ratings dataset for experimentation:**
- MovieLens Latest Small (about 100,000 ratings): https://grouplens.org/datasets/movielens/latest/
- MovieLens is not Bollywood-only. Do not combine its ratings with Bollywood metadata unless you match movie IDs reliably (for example via IMDb IDs); title-only matching can produce incorrect joins.

## Expected data layout

You may keep the original folders under `data/`, because the loader searches recursively. The following files must be present:

```text
data/
  1950-2019/
    bollywood_full.csv
  collaborative/
    ratings.csv
    links.csv
    titles.csv        # optional for this code
```

If the repository has different folders, just ensure the files are somewhere inside `data/`.

## Setup (Windows PowerShell)

Open a terminal in the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the notebook

```powershell
jupyter notebook
```

Open `notebooks/Hybrid_Bollywood_Recommendation.ipynb` and run cells from top to bottom. Set `DATA_DIR` in the first code cell if your dataset folder differs.

## Run the Streamlit UI

```powershell
streamlit run app.py
```

If the data is in a different folder, change **Data folder** in the sidebar.

## Project structure

```text
app.py                         Streamlit UI
recommender.py                 shared loading and recommendation logic
notebooks/Hybrid_Bollywood_Recommendation.ipynb
data/                          place downloaded datasets here
requirements.txt
```

## What to explain in viva

- TF-IDF converts textual metadata into weighted numerical features.
- Cosine similarity compares content vectors.
- Item-based collaborative filtering uses patterns in user ratings.
- SVD approximates the user–item rating matrix with latent factors.
- Hybrid weighting combines different recommendation signals.
- Cold-start: a new user has no rating history; content-based recommendations can still work from a selected favourite movie.
- Sparsity: most users rate only a small subset of movies.
- Evaluation should use held-out ratings. Use Precision@K, Recall@K and NDCG@K for ranking; use RMSE only when evaluating predicted numerical ratings.
- The code's SVD score is a ranking signal and should not be described as a calibrated rating or probability.

## Limitations

The quality of collaborative filtering depends on how many rating rows successfully map to the Bollywood metadata through IMDb identifiers. Check the mapped rating count in the UI. If very few ratings map, use a larger aligned rating dataset or collect ratings from classmates. Do not silently invent ratings or claim that synthetic ratings are real.
