import json

import numpy as np
import pytest

import city_clustering
import models
import preprocess
import topic_modeling
from dishes import COMMON_DISHES

# ---- предобработка ----

def test_tokenize_drops_stopwords_and_short_words():
    tokens = preprocess.tokenize("Это очень хороший и большой сервер, но он не работает")

    assert "это" not in tokens
    assert "очень" not in tokens
    assert "но" not in tokens
    assert "сервер" in tokens

def test_tokenize_lemmatizes():
    assert preprocess.tokenize("серверами серверов сервером") == ["сервер"] * 3

def test_tokenize_keeps_latin_words():
    assert "python" in preprocess.tokenize("пишем на Python и Rust")

def test_prepare_drops_too_short_documents(tmp_path):
    rows = [{"title": "Длинный материал про серверы", "summary": "сервер база данные запрос индекс"},
            {"title": "ок", "summary": ""}]
    (tmp_path / "news.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    documents, stats = preprocess.prepare("news", tmp_path, min_tokens=5)

    assert stats["loaded"] == 2
    assert stats["dropped"] == 1
    assert len(documents) == 1

def test_unknown_dataset_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        preprocess.load_records("погода", tmp_path)

def test_missing_file_names_the_problem(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess.load_records("news", tmp_path)

# ---- модели ----

CORPUS_TOKENS = [
    ["сервер", "база", "запрос", "индекс"],
    ["сервер", "база", "данные", "запрос"],
    ["кран", "груз", "стрела", "портал"],
    ["кран", "груз", "крюк", "стрела"],
    ["нейросеть", "модель", "обучение", "данные"],
    ["нейросеть", "модель", "слой", "обучение"],
]

@pytest.fixture
def corpus():
    dictionary, bow = models.build_corpus(CORPUS_TOKENS, no_below=1, no_above=0.9)
    return CORPUS_TOKENS, dictionary, bow

def test_lda_returns_requested_number_of_topics(corpus):
    tokens, dictionary, bow = corpus
    topics = models.lda_topics(models.train_lda(bow, dictionary, 3, passes=5), top_words=4)

    assert len(topics) == 3
    assert all(len(topic) == 4 for topic in topics)
    assert all(isinstance(weight, float) for topic in topics for _, weight in topic)

def test_nmf_returns_requested_number_of_topics(corpus):
    tokens, _, _ = corpus
    model, vectorizer, weights = models.train_nmf(tokens, 3)
    topics = models.nmf_topics(model, vectorizer, top_words=4)

    assert len(topics) == 3
    assert weights.shape == (len(tokens), 3)

def test_coherence_is_a_number(corpus):
    tokens, dictionary, bow = corpus
    topics = models.lda_topics(models.train_lda(bow, dictionary, 2, passes=5))

    score = models.coherence_score(topics, tokens, dictionary)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

def test_document_topic_shares_sum_to_one(corpus):
    tokens, dictionary, bow = corpus
    model = models.train_lda(bow, dictionary, 3, passes=5)

    shares = models.lda_document_topics(model, bow, 3)
    assert shares.shape == (len(tokens), 3)
    assert np.allclose(shares.sum(axis=1), 1.0, atol=1e-3)

def test_nmf_shares_are_normalized():
    weights = np.array([[2.0, 2.0], [1.0, 3.0], [0.0, 0.0]])

    shares = models.nmf_document_topics(weights)
    assert np.allclose(shares[0], [0.5, 0.5])
    assert np.allclose(shares[1], [0.25, 0.75])
    assert np.allclose(shares[2], [0.0, 0.0])   # пустая строка не делится на ноль

def test_select_topics_picks_the_best_score(corpus):
    tokens, dictionary, bow = corpus

    best, scores = models.select_topics("nmf", tokens, dictionary, bow, k_range=range(2, 5))
    assert set(scores) == {2, 3, 4}
    assert best == max(scores, key=scores.get)

# ---- CLI ----

def test_both_and_compare_mean_two_methods():
    for argv in (["--dataset", "news", "--method", "both", "--output", "x"],
                 ["--dataset", "news", "--method", "lda", "--compare", "--output", "x"]):
        args = topic_modeling.parse_args(argv)
        methods = ["lda", "nmf"] if (args.method == "both" or args.compare) else [args.method]
        assert methods == ["lda", "nmf"]

def test_topics_defaults_to_auto():
    assert topic_modeling.parse_args(["--dataset", "news", "--output", "x"]).topics == "auto"

def test_dataset_is_required():
    with pytest.raises(SystemExit):
        topic_modeling.parse_args(["--output", "x"])

# ---- кластеризация городов ----

def test_dish_frequency_is_share_of_pages():
    texts = ["тут подают эчпочмак и чак-чак", "только чак-чак", "ничего из списка", "чак-чак снова"]

    frequencies = city_clustering.dish_frequencies(texts)
    assert frequencies["чак-чак"] == 0.75
    assert frequencies["эчпочмак"] == 0.25
    assert frequencies["калитки"] == 0.0

def test_synonyms_count_as_one_dish():
    assert city_clustering.dish_frequencies(["подают треугольник"])["эчпочмак"] == 1.0

# Города с заранее известной кухней: кластеризация обязана их разложить правильно
FIXTURE = {
    "Пермь": {"посикунчики": 0.9, "пистики": 0.8, "шаньги": 0.7, "борщ": 0.9},
    "Кунгур": {"посикунчики": 0.8, "пистики": 0.9, "шаньги": 0.6, "борщ": 0.8},
    "Казань": {"эчпочмак": 0.9, "чак-чак": 0.9, "кыстыбый": 0.7, "борщ": 0.9},
    "Уфа": {"эчпочмак": 0.8, "чак-чак": 0.8, "бешбармак": 0.7, "борщ": 0.8},
    "Петрозаводск": {"калитки": 0.9, "ряпушка": 0.8, "морошка": 0.7, "борщ": 0.9},
}

@pytest.fixture
def cities():
    collected = {city: {"pages": 10, "dishes": dishes} for city, dishes in FIXTURE.items()}
    return city_clustering.build_matrix(collected)

def test_common_dishes_are_dropped(cities):
    _, dishes, _ = cities
    assert not COMMON_DISHES & set(dishes)

def test_matrix_rows_are_normalized(cities):
    _, _, matrix = cities
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

def test_three_methods_find_the_same_culinary_groups(cities):
    names, _, matrix = cities
    labels = city_clustering.cluster(matrix, 3)

    assert set(labels) == {"kmeans", "hierarchical", "spectral"}
    for method, assignment in labels.items():
        groups = {}
        for city, label in zip(names, assignment):
            groups.setdefault(label, set()).add(city)
        found = [group for group in groups.values()]
        assert {"Пермь", "Кунгур"} in found, method
        assert {"Казань", "Уфа"} in found, method
        assert {"Петрозаводск"} in found, method

def test_top_dishes_describe_the_cluster(cities):
    names, dishes, matrix = cities
    labels = city_clustering.cluster(matrix, 3)["kmeans"]

    leaders = city_clustering.top_dishes(matrix, dishes, labels, count=3)
    perm_label = labels[names.index("Пермь")]
    assert "посикунчики" in leaders[perm_label] or "пистики" in leaders[perm_label]
