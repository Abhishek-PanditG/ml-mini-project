from pathlib import Path

import streamlit as st
from recommender import load_project_data, HybridBollywoodRecommender

st.set_page_config(
    page_title="Bollywood Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0b1020 0%, #151b31 100%);
    }
    h1, h2, h3, p, label { color: #f4f6fb; }
    div[data-testid="stDataFrame"] {
        border: 1px solid #303954;
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

@st.cache_resource(show_spinner="Loading movies...")
def build_model(data_dir: str):
    movies, ratings = load_project_data(data_dir)
    return HybridBollywoodRecommender(movies, ratings), movies

st.title("🎬 Type a movie you love")

try:
    model, movies = build_model(str(DATA_DIR))
except Exception as exc:
    st.error(f"Could not load the movie dataset: {exc}")
    st.stop()

movie_titles = sorted(movies["title"].dropna().astype(str).unique().tolist())
selected_title = st.selectbox(
    "Type a movie you love",
    options=movie_titles,
    index=None,
    placeholder="Start typing a movie title...",
    label_visibility="collapsed",
)

if selected_title:
    recommendations = model.recommend_content(selected_title, n=10)
    st.subheader("10 movies you may like")

    if recommendations.empty:
        st.info("No recommendations found. Try another movie.")
    else:
        display_cols = [
            col for col in ["title", "year", "genres", "rating"]
            if col in recommendations.columns
        ]
        recommendations = recommendations[display_cols].rename(
            columns={
                "title": "Movie",
                "year": "Year",
                "genres": "Genres",
                "rating": "IMDb rating",
            }
        )
        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True,
            height=385,
        )
