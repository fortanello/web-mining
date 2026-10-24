"""Тематическая группировка роликов."""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

MAX_FEATURES = 5000
RANDOM_STATE = 42
TOP_WORDS = 10


# Текст ролика собирается из всех источников сразу: название и описание дают тему,
# транскрипт — содержание, названия найденных объектов работают как обычные слова
def document_text(meta, transcript, detections):
    parts = [str(meta.get("title") or ""), str(meta.get("description") or "")]
    parts.extend(str(tag) for tag in (meta.get("tags") or []))
    parts.extend(segment["text"] for segment in transcript.get("segments", []))
    parts.extend(detections.get("stats", {}).get("by_label", {}))
    return " ".join(parts).lower()


def build_matrix(texts, max_features=MAX_FEATURES):
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=1)
    return vectorizer, vectorizer.fit_transform(texts)


# Кластеризация осмысленна от трёх роликов: на одном-двух делить нечего,
# а silhouette вообще не определён, когда кластеров столько же, сколько объектов
def cluster(matrix, clusters):
    samples = matrix.shape[0]
    if samples < 3:
        return np.zeros(samples, dtype=int), 1, 0.0

    clusters = max(2, min(clusters, samples - 1))
    labels = KMeans(n_clusters=clusters, random_state=RANDOM_STATE,
                    n_init=10).fit_predict(matrix)

    unique = len(set(labels))
    score = float(silhouette_score(matrix, labels)) if 1 < unique < samples else 0.0
    return labels, clusters, score


def nmf_topics(matrix, vectorizer, topics, top_words=TOP_WORDS):
    topics = max(1, min(topics, matrix.shape[0]))
    model = NMF(n_components=topics, random_state=RANDOM_STATE,
                init="nndsvda", max_iter=1000)
    weights = model.fit_transform(matrix)
    names = vectorizer.get_feature_names_out()

    found = []
    for row in model.components_:
        top = row.argsort()[:-top_words - 1:-1]
        found.append([{"word": names[index], "weight": round(float(row[index]), 4)}
                      for index in top])
    return found, weights


# Слова, по которым кластер отличается от остальных
def cluster_words(matrix, vectorizer, labels, top_words=TOP_WORDS):
    names = vectorizer.get_feature_names_out()
    dense = matrix.toarray()
    words = {}
    for label in sorted(set(labels)):
        mean = dense[labels == label].mean(axis=0)
        top = mean.argsort()[::-1][:top_words]
        words[int(label)] = [names[index] for index in top if mean[index] > 0]
    return words


def group(videos, root, clusters=5, force=False):
    path = Path(root) / "topics.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    texts = [video["text"] for video in videos]
    vectorizer, matrix = build_matrix(texts)

    labels, used, score = cluster(matrix, clusters)
    topics, weights = nmf_topics(matrix, vectorizer, used)

    result = {
        "videos": [{"id": video["id"], "title": video["title"], "cluster": int(label),
                    "topic": int(weights[index].argmax())}
                   for index, (video, label) in enumerate(zip(videos, labels))],
        "clusters": used,
        "silhouette": round(score, 4),
        "cluster_words": cluster_words(matrix, vectorizer, labels),
        "topics": topics,
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  кластеров {used}, silhouette {score:.3f}")
    return result
