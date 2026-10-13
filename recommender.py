from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")

def _read_csv(path):
    return pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

def _find_file(data_dir, filename):
    hits = list(Path(data_dir).rglob(filename))
    return hits[0] if hits else None

def _first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _id_key(x):
    if pd.isna(x):
        return ""
    # Handles numeric IMDb IDs and values such as tt0133093.
    digits = re.sub(r"\D", "", str(x))
    return digits.lstrip("0") or digits

def load_project_data(data_dir="data"):
    """
    Expected source files:
      - bollywood_full.csv (Bollywood metadata)
      - ratings.csv, links.csv, titles.csv (collaborative rating data)
    The loader searches subfolders recursively.
    """
    meta_path = _find_file(data_dir, "bollywood_full.csv")
    ratings_path = _find_file(data_dir, "ratings.csv")
    links_path = _find_file(data_dir, "links.csv")
    titles_path = _find_file(data_dir, "titles.csv")

    if not meta_path:
        raise FileNotFoundError(
            "Missing bollywood_full.csv. Download the TIMDB repository dataset "
            "and place its 1950-2019/bollywood_full.csv file inside the data folder."
        )
    if not all([ratings_path, links_path]):
        raise FileNotFoundError(
            "Missing collaborative ratings files. Place ratings.csv and links.csv "
            "from the TIMDB repository inside data/collaborative/ (or anywhere under data/)."
        )

    movies = _read_csv(meta_path)
    movies.columns = [_norm_col(c) for c in movies.columns]
    ratings = _read_csv(ratings_path)
    ratings.columns = [_norm_col(c) for c in ratings.columns]
    links = _read_csv(links_path)
    links.columns = [_norm_col(c) for c in links.columns]

    # Normalize metadata schema
    title_col = _first_existing(movies, ["title", "name", "movie_title", "title_imdb", "original_title"])
    imdb_col = _first_existing(movies, ["imdb_id", "imdbid", "imdb"])
    if not title_col or not imdb_col:
        raise ValueError(
            f"Could not identify title/IMDb columns in metadata. Available columns: {list(movies.columns)}"
        )
    movies = movies.rename(columns={title_col: "title", imdb_col: "imdb_id"})
    movies["imdb_key"] = movies["imdb_id"].map(_id_key)

    # Standardize optional metadata columns
    alias_map = {
        "year": ["year_of_release", "year", "release_year", "release_date"],
        "genres": ["genres", "genre"],
        "overview": ["summary", "overview", "story", "plot", "synopsis", "description"],
        "actors": ["actors", "cast", "star", "stars"],
        "directors": ["directors", "director", "crew"],
        "rating": ["imdb_rating", "rating", "imdb_score", "score"],
        "poster": ["poster_path", "poster_url", "poster"],
    }
    for target, candidates in alias_map.items():
        col = _first_existing(movies, candidates)
        if col and col != target:
            movies = movies.rename(columns={col: target})
        if target not in movies.columns:
            movies[target] = ""

    movies["title"] = movies["title"].astype(str).str.strip()
    movies = movies[movies["title"].ne("") & movies["title"].ne("nan")].copy()
    movies = movies.drop_duplicates(subset=["imdb_key"], keep="first")
    movies["content_text"] = (
        movies["title"].fillna("").astype(str) + " " +
        movies["genres"].fillna("").astype(str) + " " +
        movies["directors"].fillna("").astype(str) + " " +
        movies["actors"].fillna("").astype(str) + " " +
        movies["overview"].fillna("").astype(str)
    ).str.lower()

    # Normalize rating/link schemas
    movie_id_col = _first_existing(links, ["movie_id", "movieid"])
    link_imdb_col = _first_existing(links, ["imdb_id", "imdbid"])
    if not movie_id_col or not link_imdb_col:
        raise ValueError(f"Could not identify movie_id/imdb_id in links.csv: {list(links.columns)}")
    links = links.rename(columns={movie_id_col: "movie_id", link_imdb_col: "imdb_id"})
    links["imdb_key"] = links["imdb_id"].map(_id_key)

    user_col = _first_existing(ratings, ["user_id", "userid", "user"])
    rating_movie_col = _first_existing(ratings, ["movie_id", "movieid", "item_id"])
    rating_col = _first_existing(ratings, ["rating", "score"])
    if not user_col or not rating_movie_col or not rating_col:
        raise ValueError(f"Could not identify user/movie/rating columns in ratings.csv: {list(ratings.columns)}")
    ratings = ratings.rename(columns={user_col: "user_id", rating_movie_col: "movie_id", rating_col: "rating"})
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating", "user_id", "movie_id"])
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["movie_id"] = ratings["movie_id"].astype(str)
    links["movie_id"] = links["movie_id"].astype(str)

    rating_links = ratings.merge(links[["movie_id", "imdb_key"]], on="movie_id", how="inner")
    rating_links = rating_links.merge(
        movies[["imdb_key", "title"]].drop_duplicates("imdb_key"),
        on="imdb_key", how="inner"
    )
    rating_links = rating_links[["user_id", "imdb_key", "rating"]].drop_duplicates(
        ["user_id", "imdb_key"], keep="last"
    )
    # Internal stable movie key based on IMDb identifier.
    movies = movies[movies["imdb_key"].ne("")].copy()
    movies["movie_key"] = movies["imdb_key"]
    rating_links = rating_links.rename(columns={"imdb_key": "movie_key"})
    rating_links = rating_links[rating_links["movie_key"].isin(movies["movie_key"])]

    return movies.reset_index(drop=True), rating_links.reset_index(drop=True)

class HybridBollywoodRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        self.movies["movie_key"] = self.movies["movie_key"].astype(str)
        self.ratings["movie_key"] = self.ratings["movie_key"].astype(str)
        self.movie_index = {key: i for i, key in enumerate(self.movies["movie_key"])}
        self.title_to_key = {}
        for _, row in self.movies.iterrows():
            self.title_to_key.setdefault(str(row["title"]), str(row["movie_key"]))
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
        self.content_matrix = self.vectorizer.fit_transform(self.movies["content_text"].fillna(""))
        self.content_similarity = cosine_similarity(self.content_matrix, dense_output=True)
        self.rating_matrix = None
        self.user_ids = []
        self.item_keys = []
        self.item_similarity = None
        self.svd_scores = None
        self.svd_user_index = {}
        self.svd_item_index = {}
        self._fit_collaborative()

    def _fit_collaborative(self):
        valid = self.ratings[self.ratings["movie_key"].isin(self.movie_index)]
        if valid.empty:
            return
        pivot = valid.pivot_table(index="user_id", columns="movie_key", values="rating", aggfunc="mean")
        # Keep only catalogue items with at least one rating and retain a consistent ordering.
        self.user_ids = pivot.index.astype(str).tolist()
        self.item_keys = pivot.columns.astype(str).tolist()
        self.rating_matrix = pivot.fillna(0.0).to_numpy(dtype=float)
        self.svd_user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.svd_item_index = {m: i for i, m in enumerate(self.item_keys)}
        item_user = self.rating_matrix.T
        self.item_similarity = cosine_similarity(item_user)
        n_components = min(20, max(1, min(self.rating_matrix.shape) - 1))
        if min(self.rating_matrix.shape) >= 2:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(self.rating_matrix)
            self.svd_scores = user_factors @ svd.components_
        else:
            self.svd_scores = self.rating_matrix.copy()

    def find_movie_key(self, title):
        if title in self.title_to_key:
            return self.title_to_key[title]
        match = self.movies[self.movies["title"].str.casefold() == str(title).casefold()]
        return str(match.iloc[0]["movie_key"]) if not match.empty else None

    def recommend_content(self, title, n=10):
        key = self.find_movie_key(title)
        if key not in self.movie_index:
            return pd.DataFrame()
        idx = self.movie_index[key]
        scores = self.content_similarity[idx]
        out = self.movies.copy()
        out["content_score"] = scores
        out = out[out["movie_key"] != key]
        return out.sort_values("content_score", ascending=False).head(n)

    def _normalise(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        lo, hi = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    def recommend_user(self, user_id, seed_title=None, n=10, content_weight=0.4, item_weight=0.3, svd_weight=0.3):
        # Content score is based on the selected seed movie. If none is supplied, it is zero.
        content_scores = np.zeros(len(self.movies), dtype=float)
        seed_key = self.find_movie_key(seed_title) if seed_title else None
        if seed_key in self.movie_index:
            content_scores = self.content_similarity[self.movie_index[seed_key]].copy()

        item_scores = np.zeros(len(self.movies), dtype=float)
        svd_scores = np.zeros(len(self.movies), dtype=float)
        user_id = str(user_id)
        if self.rating_matrix is not None and user_id in self.svd_user_index:
            uidx = self.svd_user_index[user_id]
            user_ratings = self.rating_matrix[uidx]
            # Item-item weighted sum of the user's observed ratings.
            denom = np.abs(user_ratings) @ np.abs(self.item_similarity) + 1e-9
            raw_item = (user_ratings @ self.item_similarity) / denom
            for j, key in enumerate(self.item_keys):
                if key in self.movie_index:
                    item_scores[self.movie_index[key]] = raw_item[j]
            raw_svd = self.svd_scores[uidx]
            for j, key in enumerate(self.item_keys):
                if key in self.movie_index:
                    svd_scores[self.movie_index[key]] = raw_svd[j]
            # Hide movies already rated by this user.
            rated = {self.item_keys[j] for j, val in enumerate(user_ratings) if val > 0}
        else:
            rated = set()

        c = self._normalise(content_scores)
        i = self._normalise(item_scores)
        s = self._normalise(svd_scores)
        total_weight = max(1e-9, content_weight + item_weight + svd_weight)
        hybrid = (content_weight*c + item_weight*i + svd_weight*s) / total_weight
        out = self.movies.copy()
        out["content_score"] = c
        out["item_cf_score"] = i
        out["svd_score"] = s
        out["hybrid_score"] = hybrid
        if seed_key:
            out = out[out["movie_key"] != seed_key]
        out = out[~out["movie_key"].isin(rated)]
        return out.sort_values("hybrid_score", ascending=False).head(n)

def top_content_examples(model, title, n=10):
    cols = [c for c in ["title", "year", "genres", "rating", "content_score"] if c in model.movies.columns or c == "content_score"]
    return model.recommend_content(title, n)[cols]
