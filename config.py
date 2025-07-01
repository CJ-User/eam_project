from collections import namedtuple

DataConfig = namedtuple("DataConfig", [
    "DATA_PATH", "ITEM_PATH", "ITEM_WITH_DIRECTOR_AND_ACTOR_PATH", "DATA_COLUMNS", "ITEM_COLUMNS"
])

DATA_CONFIG = DataConfig(
    DATA_PATH="./data/ml-100k/u.data",
    ITEM_PATH="./data/ml-100k/u.item",
    ITEM_WITH_DIRECTOR_AND_ACTOR_PATH='./data/ml-100k/u_with_directors_and_actors.item',
    DATA_COLUMNS=["user_id", "item_id", "rating", "timestamp"],
    ITEM_COLUMNS=[
        "movie_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
)

MMFConfig = namedtuple("MMFConfig", [
    "USE_CACHE", "MODEL_PATH", "LATENT_K", "ALPHA",
    "BETA", "EPOCHS", "THETA_CLIP", "OMEGA_CLIP"
])

MMF_CONFIG = MMFConfig(
    USE_CACHE=False,
    MODEL_PATH="saved_models/mmf_model.pkl",
    LATENT_K=30,
    ALPHA=0.02,
    BETA=0.02,
    EPOCHS=150,
    THETA_CLIP=[0.05, 4.0],
    OMEGA_CLIP=[0.05, 4.0]
)
