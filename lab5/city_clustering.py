"""Кластеризация городов по кулинарным предпочтениям (Пример 2 из лекции 6).

Города кодируются векторами частотности блюд, собранной со страниц Википедии
о кухне региона, и кластеризуются тремя методами.

    python city_clustering.py --collect
    python city_clustering.py --clusters 3 --output cities
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from dishes import COMMON_DISHES, DISHES

CITIES = ["Пермь", "Кунгур", "Казань", "Уфа", "Москва", "Санкт-Петербург",
          "Петрозаводск", "Новосибирск", "Хабаровск", "Владивосток"]

API = "https://ru.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "web-mining-lab5/1.0 (educational project; contact: student@example.com)"}
PAGES_PER_CITY = 12
PAUSE = 0.5

DATA_PATH = Path(__file__).resolve().parent / "data" / "city_dishes.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def api_request(params):
    params = {**params, "format": "json", "formatversion": "2"}
    response = requests.get(API, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


# Страницы о кухне города: берём выдачу поиска Википедии
def city_pages(city, limit=PAGES_PER_CITY):
    found = api_request({"action": "query", "list": "search",
                         "srsearch": f"кухня {city}", "srlimit": limit})
    return [row["title"] for row in found.get("query", {}).get("search", [])]


def page_texts(titles):
    texts = []
    for start in range(0, len(titles), 10):
        batch = titles[start:start + 10]
        data = api_request({"action": "query", "prop": "extracts", "explaintext": "1",
                            "titles": "|".join(batch)})
        for page in data.get("query", {}).get("pages", []):
            texts.append(page.get("extract", "") or "")
        time.sleep(PAUSE)
    return texts


# Частотность блюда = доля страниц города, где оно упомянуто. Всегда в [0, 1]
def dish_frequencies(texts):
    frequencies = {}
    for dish, synonyms in DISHES.items():
        pattern = re.compile("|".join(re.escape(word) for word in synonyms), re.IGNORECASE)
        hits = sum(1 for text in texts if pattern.search(text))
        frequencies[dish] = hits / len(texts) if texts else 0.0
    return frequencies


def collect(cities, path=DATA_PATH):
    collected = {}
    for city in cities:
        titles = city_pages(city)
        texts = page_texts(titles)
        collected[city] = {"pages": len(texts), "dishes": dish_frequencies(texts)}
        found = sum(1 for value in collected[city]["dishes"].values() if value > 0)
        print(f"  {city:18} страниц {len(texts):2}, блюд найдено {found}")
        if len(texts) < 5:
            print(f"    предупреждение: мало страниц, статистика будет шумной")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {path}")
    return collected


# Матрица город на блюдо. L2-нормализация нужна, чтобы сравнивать профили,
# а не объёмы: город с большим числом страниц иначе окажется ближе ко всем
def build_matrix(collected, drop_common=True, min_variance=0.0):
    # ключи со знаком подчёркивания служебные, городами не являются
    cities = [city for city in collected if not city.startswith("_")]
    dishes = [dish for dish in DISHES if not (drop_common and dish in COMMON_DISHES)]

    matrix = np.array([[collected[city]["dishes"].get(dish, 0.0) for dish in dishes]
                       for city in cities])

    if min_variance > 0:
        keep = matrix.var(axis=0) > min_variance
        matrix = matrix[:, keep]
        dishes = [dish for dish, flag in zip(dishes, keep) if flag]

    return cities, dishes, normalize(matrix)


def cluster(matrix, clusters):
    return {
        "kmeans": KMeans(n_clusters=clusters, random_state=42,
                         n_init=10).fit_predict(matrix),
        "hierarchical": AgglomerativeClustering(n_clusters=clusters,
                                                linkage="ward").fit_predict(matrix),
        "spectral": SpectralClustering(n_clusters=clusters, affinity="cosine",
                                       random_state=42).fit_predict(matrix),
    }


# Метод локтя и silhouette: формальное подтверждение выбранного числа кластеров
def evaluate_clusters(matrix, k_range):
    inertia, silhouette = {}, {}
    for clusters in k_range:
        model = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit(matrix)
        inertia[clusters] = float(model.inertia_)
        silhouette[clusters] = float(silhouette_score(matrix, model.labels_))
    return inertia, silhouette


# Номера кластеров у методов произвольны, поэтому разбиения сравниваются
# индексом Rand: 1.0 — полное совпадение, 0.0 — совпадение на уровне случайного
def agreement(labels):
    methods = sorted(labels)
    return {f"{first} и {second}": float(adjusted_rand_score(labels[first], labels[second]))
            for index, first in enumerate(methods) for second in methods[index + 1:]}


def top_dishes(matrix, dishes, labels, count=5):
    result = {}
    for label in sorted(set(labels)):
        rows = matrix[labels == label]
        mean = rows.mean(axis=0)
        best = mean.argsort()[::-1][:count]
        result[int(label)] = [dishes[index] for index in best if mean[index] > 0]
    return result


def plot_clusters(matrix, cities, labels, path):
    points = PCA(n_components=2, random_state=42).fit_transform(matrix)

    figure, axis = plt.subplots(figsize=(10, 7.5))
    axis.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=180, edgecolor="black")

    # Города со схожей кухней ложатся почти в одну точку — разводим подписи по вертикали
    taken = {}
    for index, city in enumerate(cities):
        spot = (round(points[index, 0], 1), round(points[index, 1], 1))
        shift = taken.get(spot, 0)
        taken[spot] = shift + 1
        axis.annotate(city, (points[index, 0], points[index, 1]),
                      textcoords="offset points", xytext=(14, 10 - shift * 20),
                      fontsize=11,
                      arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.6)
                      if shift else None)

    axis.set_title("Города в пространстве блюд (проекция PCA)")
    axis.set_xlabel("Главная компонента 1")
    axis.set_ylabel("Главная компонента 2")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_evaluation(inertia, silhouette, path):
    keys = sorted(inertia)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.plot(keys, [inertia[key] for key in keys], marker="o", color="#2b7bba")
    left.set_title("Метод локтя")
    left.set_xlabel("Число кластеров")
    left.set_ylabel("Инерция")
    left.grid(alpha=0.3)

    best = max(silhouette, key=silhouette.get)
    right.plot(keys, [silhouette[key] for key in keys], marker="o", color="#2b7bba")
    right.axvline(best, color="#c0392b", linestyle="--",
                  label=f"максимум при k = {best}")
    right.set_title("Silhouette score")
    right.set_xlabel("Число кластеров")
    right.grid(alpha=0.3)
    right.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="city_clustering",
        description="Кластеризация городов по упоминаемости блюд.")
    parser.add_argument("--collect", action="store_true",
                        help="собрать данные из Википедии и сохранить в data/")
    parser.add_argument("--cities", nargs="+", default=CITIES, help="список городов")
    parser.add_argument("--clusters", type=int, default=3, help="число кластеров")
    parser.add_argument("--output", default="cities", help="префикс файлов результата")
    parser.add_argument("--data", default=str(DATA_PATH), help="файл с собранными данными")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = Path(args.data)

    if args.collect:
        collect(args.cities, path)
        return 0

    if not path.exists():
        print(f"Нет файла {path}. Сначала соберите данные: "
              f"python city_clustering.py --collect")
        return 1

    collected = json.loads(path.read_text(encoding="utf-8"))
    if collected.get("_demo"):
        print("ВНИМАНИЕ: это демонстрационный набор, а не собранные данные.")
        print("Настоящие данные: python city_clustering.py --collect")

    cities, dishes, matrix = build_matrix(collected)
    print(f"Городов: {len(cities)}, измерений (блюд): {len(dishes)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    prefix = OUTPUT_DIR / args.output

    labels = cluster(matrix, args.clusters)
    for method, assignment in labels.items():
        groups = {}
        for city, label in zip(cities, assignment):
            groups.setdefault(int(label), []).append(city)
        print(f"  {method:13}: " + " | ".join(
            f"[{label}] {', '.join(members)}" for label, members in sorted(groups.items())))

    # Устойчивость: если три метода согласны, структура в данных есть
    scores = agreement(labels)
    for pair, score in scores.items():
        print(f"  согласие {pair}: {score:.2f}")
    if min(scores.values()) > 0.8:
        print("  методы согласны — разбиение устойчивое")
    else:
        print("  методы расходятся — структура в данных слабая, результат ненадёжен")

    upper = min(len(cities) - 1, 8)
    inertia, silhouette = evaluate_clusters(matrix, range(2, upper + 1))
    print(f"  silhouette максимален при k = {max(silhouette, key=silhouette.get)}")

    plot_clusters(matrix, cities, labels["kmeans"], f"{prefix}_clusters.png")
    plot_evaluation(inertia, silhouette, f"{prefix}_evaluation.png")

    leaders = top_dishes(matrix, dishes, labels["kmeans"])
    for label, items in leaders.items():
        print(f"  топ блюд кластера {label}: {', '.join(items)}")

    report = {
        "cities": cities,
        "dishes": dishes,
        "clusters": {method: [int(value) for value in assignment]
                     for method, assignment in labels.items()},
        "top_dishes": leaders,
        "agreement": scores,
        "inertia": inertia,
        "silhouette": silhouette,
    }
    Path(f"{prefix}_clusters.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Результаты: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
