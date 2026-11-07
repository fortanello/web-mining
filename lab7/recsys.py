import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

from models import MODELS, RELEVANT_RATING, SEED, Recommenders


LAB = Path(__file__).resolve().parent
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ALL = "все"
SEGMENTS = (ALL, "0", "1–19", "20+")
REFERENCE_K = 10


def load_data(archive):
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=archive.parent) as directory:
            temporary = Path(directory) / archive.name
            urlretrieve(DATA_URL, temporary)
            with ZipFile(temporary) as data:
                if data.testzip() is not None:
                    raise ValueError("Скачанный архив MovieLens повреждён")
            temporary.replace(archive)

    with ZipFile(archive) as data:
        with data.open("ml-latest-small/ratings.csv") as handle:
            ratings = pd.read_csv(handle)
        with data.open("ml-latest-small/movies.csv") as handle:
            movies = pd.read_csv(handle).set_index("movieId").sort_index()
    return ratings, movies


def split_ratings(ratings):
    ordered = ratings.sort_values("timestamp", kind="stable")
    cutoff = int(ordered.iloc[int(len(ordered) * 0.8)]["timestamp"])
    train = ordered[ordered["timestamp"] < cutoff]
    test = ordered[ordered["timestamp"] >= cutoff]
    return train, test, cutoff


def segment(history_size):
    if history_size == 0:
        return "0"
    if history_size < 20:
        return "1–19"
    return "20+"


def paired_intervals(ranking, model_names):
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(ranking), size=(2000, len(ranking)))
    intervals = []
    for model in model_names:
        if model == "popular":
            continue
        difference = (ranking[model] - ranking["popular"]).to_numpy()
        low, high = np.percentile(difference[samples].mean(axis=1), [2.5, 97.5])
        intervals.append({"model": model, "difference": float(difference.mean()),
                          "low": float(low), "high": float(high)})
    return intervals


def ranking_metrics(recs, relevant, k):
    hits = 0
    average_precision = 0
    dcg = 0
    for rank, movie in enumerate(recs[:k], start=1):
        if movie in relevant:
            hits += 1
            average_precision += hits / rank
            dcg += 1 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return hits / k, hits / len(relevant), average_precision / ideal_hits, dcg / ideal_dcg


def genre_diversity(recs, genres):
    distances = [1 - len(genres[a] & genres[b]) / len(genres[a] | genres[b])
                 for a, b in combinations(recs, 2) if genres[a] and genres[b]]
    return float(np.mean(distances)) if distances else np.nan


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(system, train, test, movies, model_names, ks, output, user_id):
    train_sizes = train.groupby("userId").size().to_dict()
    candidate_ids = set(movies.index)
    positives = test[test["rating"] >= RELEVANT_RATING]
    relevant = {}
    for user, group in positives.groupby("userId"):
        eligible = set(group["movieId"]) & candidate_ids
        eligible -= system.seen.get(user, set())
        if eligible:
            relevant[user] = eligible

    if not relevant:
        raise ValueError("В тестовой части нет релевантных фильмов из общего каталога")

    genres = {movie: set(value.split("|")) - {"(no genres listed)"}
              for movie, value in movies["genres"].items()}
    popular_count = max(1, int(np.ceil(0.1 * len(candidate_ids))))
    head = set(system.movie_ids[:popular_count])
    largest_k = max(ks)
    recommendations = {}
    rows = []
    by_user = []
    for model in model_names:
        recommendations[model] = {user: system.recommend(user, model, largest_k) for user in relevant}
        for k in ks:
            for user, recs in recommendations[model].items():
                precision, recall, ap, ndcg = ranking_metrics(recs, relevant[user], k)
                shown = recs[:k]
                by_user.append({"model": model, "k": k, "user": user,
                                "segment": segment(train_sizes.get(user, 0)),
                                "precision": precision, "recall": recall, "map": ap, "ndcg": ndcg,
                                "diversity": genre_diversity(shown, genres),
                                "tail_share": sum(movie not in head for movie in shown) / len(shown)
                                if shown else np.nan})

    individual = pd.DataFrame(by_user)
    for (model, k), group in individual.groupby(["model", "k"], sort=False):
        for label in SEGMENTS:
            part = group if label == ALL else group[group["segment"] == label]
            if part.empty:
                continue
            unique = {movie for user in part["user"] for movie in recommendations[model][user][:k]}
            averages = part[["precision", "recall", "map", "ndcg", "diversity", "tail_share"]].mean()
            rows.append({"model": model, "segment": label, "k": k, "users": len(part),
                         **averages.to_dict(), "coverage": len(unique) / len(candidate_ids)})

    results = pd.DataFrame(rows)
    results.to_csv(output / "metrics.csv", index=False, float_format="%.6f")

    reference_k = REFERENCE_K if REFERENCE_K in ks else largest_k
    ranking = individual[individual["k"] == reference_k].pivot(index="user", columns="model", values="ndcg")
    examples = {"первый пользователь в оценке": int(ranking.index[0])}
    intervals = {}
    if "popular" in model_names and len(model_names) > 1:
        intervals[ALL] = paired_intervals(ranking, model_names)
        active = ranking.loc[[user for user in ranking.index if train_sizes.get(user, 0) >= 20]]
        if len(active) >= 2:
            intervals["20+"] = paired_intervals(active, model_names)
        if "als" in model_names:
            difference = ranking["als"] - ranking["popular"]
            examples = {"наибольшая разница в пользу ALS": int(difference.idxmax()),
                        "наибольшая разница в пользу бейзлайна": int(difference.idxmin())}

    if user_id is not None:
        examples["выбранный пользователь"] = user_id

    summary = {
        "test_users": int(test["userId"].nunique()),
        "test_users_with_positives": int(positives["userId"].nunique()),
        "evaluated_users": len(relevant),
        "test_positives": len(positives),
        "eligible_positives": sum(map(len, relevant.values())),
        "cold_item_positives": int((~positives["movieId"].isin(candidate_ids)).sum()),
        "repeat_positives": sum(movie in system.seen.get(user, set())
                                for user, movie in positives[["userId", "movieId"]].itertuples(index=False)),
        "segments": {label: sum(segment(train_sizes.get(user, 0)) == label for user in relevant)
                     for label in SEGMENTS[1:]},
        "reference_k": reference_k,
        "bootstrap_ndcg": intervals,
    }
    target_sizes = pd.DataFrame({
        "segment": [segment(train_sizes.get(user, 0)) for user in relevant],
        "relevant": [len(target) for target in relevant.values()],
    })
    summary["median_eligible_positives"] = {
        ALL: float(target_sizes["relevant"].median()),
        **target_sizes.groupby("segment")["relevant"].median().to_dict(),
    }
    records = example_records(system, train, movies, model_names, examples, reference_k, relevant)
    write_json(output / "recs.json", {"k": reference_k, "examples": records})
    return results, summary


def example_records(system, train, movies, model_names, examples, k, relevant):
    def info(movie):
        return {"movie_id": int(movie), "title": movies.at[movie, "title"],
                "genres": movies.at[movie, "genres"].split("|")}

    records = []
    for reason, user in examples.items():
        history = train[(train["userId"] == user) & (train["rating"] >= RELEVANT_RATING)]
        target = relevant.get(user, set())
        rankings = {}
        for model in model_names:
            recs = system.recommend(user, model, k)
            rankings[model] = {"items": [{**info(movie), "hit": movie in target}
                                         for movie in recs],
                               "ndcg": ranking_metrics(recs, target, k)[3] if target else None}
        records.append({"user_id": user, "reason": reason,
                        "train_ratings": len(system.seen.get(user, set())),
                        "liked_train": [{**info(movie), "rating": rating}
                                         for movie, rating in history[["movieId", "rating"]].itertuples(index=False)],
                        "relevant_test": [info(movie) for movie in sorted(target)],
                        "rankings": rankings})
    return records


def main():
    parser = argparse.ArgumentParser(description="Сравнение рекомендательных моделей на MovieLens")
    parser.add_argument("--dataset", choices=["movielens"], default="movielens")
    parser.add_argument("--data", type=Path, default=LAB / "data" / "ml-latest-small.zip")
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output", type=Path, default=LAB / "results")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--user-id", type=int)
    args = parser.parse_args()
    if min(args.k) <= 0:
        parser.error("K должно быть положительным")
    if not args.evaluate and args.user_id is None:
        parser.error("укажите --evaluate или --user-id")

    ratings, movies = load_data(args.data)
    train, test, cutoff = split_ratings(ratings)
    candidates = movies.loc[movies.index.isin(train["movieId"])]
    model_names = MODELS if args.model == "all" else (args.model,)
    system = Recommenders(train, candidates, model_names)
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.evaluate:
        records = example_records(system, train, candidates, model_names,
                                  {"выбранный пользователь": args.user_id}, max(args.k), {})
        write_json(args.output / "recs.json", {"k": max(args.k), "examples": records})
        print(f"Рекомендации сохранены в {args.output / 'recs.json'}")
        return

    from plots import plot_results

    results, summary = evaluate(system, train, test, candidates, model_names, sorted(set(args.k)),
                                args.output, args.user_id)
    summary.update({
        "ratings": len(ratings), "users": int(ratings["userId"].nunique()),
        "movies": len(movies), "train_ratings": len(train), "test_ratings": len(test),
        "train_users": int(train["userId"].nunique()), "candidate_movies": len(candidates),
        "cutoff_utc": datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(),
        "sparsity": 1 - len(ratings) / (ratings["userId"].nunique() * len(movies)),
        "train_positives": int((train["rating"] >= RELEVANT_RATING).sum()),
        "activity": {str(quantile): value for quantile, value in
                     ratings.groupby("userId").size().quantile([0.25, 0.5, 0.75, 1]).items()},
        "archive_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    })
    write_json(args.output / "summary.json", summary)
    plot_results(ratings, results, summary, args.output)
    print(results[(results["k"] == summary["reference_k"]) & (results["segment"] == ALL)]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Результаты сохранены в {args.output}")


if __name__ == "__main__":
    main()
