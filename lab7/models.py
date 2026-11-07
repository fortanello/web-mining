import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import Dataset, Reader, SVD


SEED = 42
RELEVANT_RATING = 4
MODELS = ("popular", "content", "svd", "als")


class Recommenders:
    def __init__(self, train, movies, model_names):
        # Единый порядок кандидатов. При равном балле первым идёт более популярный фильм.
        positive = train[train["rating"] >= RELEVANT_RATING]
        positive_counts = positive["movieId"].value_counts().reindex(movies.index, fill_value=0)
        self.movie_ids = positive_counts.sort_values(ascending=False, kind="stable").index.to_numpy()
        self.movie_positions = {movie: i for i, movie in enumerate(self.movie_ids)}
        self.seen = train.groupby("userId")["movieId"].agg(set).to_dict()
        self.liked = positive.groupby("userId")["movieId"].agg(list).to_dict()
        self.popular = -np.arange(len(self.movie_ids), dtype=float)

        if "content" in model_names:
            catalog = movies.loc[self.movie_ids]
            titles = catalog["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
            genres = catalog["genres"].str.replace("(no genres listed)", "", regex=False)
            texts = titles + " " + genres.str.replace("|", " ", regex=False)
            self.content = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b").fit_transform(texts)

        if "svd" in model_names:
            data = Dataset.load_from_df(train[["userId", "movieId", "rating"]],
                                        Reader(rating_scale=(0.5, 5)))
            self.svd_train = data.build_full_trainset()
            self.svd = SVD(n_factors=50, n_epochs=20, random_state=SEED).fit(self.svd_train)
            self.svd_items = [self.svd_train.to_inner_iid(movie) for movie in self.movie_ids]

        if "als" in model_names:
            user_rows = {user: i for i, user in enumerate(self.seen)}
            rows = positive["userId"].map(user_rows)
            columns = positive["movieId"].map(self.movie_positions)
            matrix = csr_matrix((np.ones(len(positive), dtype=np.float32), (rows, columns)),
                                shape=(len(user_rows), len(self.movie_ids)))
            als = AlternatingLeastSquares(factors=32, iterations=15, random_state=SEED,
                                          use_gpu=False, num_threads=1)
            als.fit(matrix, show_progress=False)
            self.als_items = normalize(als.item_factors)
            self.als_known = positive_counts.loc[self.movie_ids].to_numpy() > 0

    def recommend(self, user, model, k):
        seen = self.seen.get(user, set())
        liked = self.liked.get(user, [])
        liked_positions = [self.movie_positions[movie] for movie in liked]
        scores = self.popular

        if model == "content" and liked:
            profile = np.asarray(self.content[liked_positions].mean(axis=0)).ravel()
            # Норма профиля одинакова для всех кандидатов, поэтому порядок сохраняется.
            scores = np.asarray(self.content @ profile).ravel()
        elif model == "svd" and seen:
            user_position = self.svd_train.to_inner_uid(user)
            # Средний рейтинг и смещение пользователя одинаковы для всех кандидатов.
            scores = self.svd.bi[self.svd_items] + self.svd.qi[self.svd_items] @ self.svd.pu[user_position]
        elif model == "als" and liked:
            # Среднее косинусное сходство с понравившимися фильмами.
            profile = self.als_items[liked_positions].mean(axis=0)
            scores = self.als_items @ profile
            scores[~self.als_known] = -np.inf

        order = np.argsort(-scores, kind="stable")
        return [int(self.movie_ids[i]) for i in order if self.movie_ids[i] not in seen][:k]
