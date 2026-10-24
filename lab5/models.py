"""Тематические модели: LDA через gensim, NMF через sklearn, оценка по coherence."""
import numpy as np
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42
LDA_PASSES = 15
MAX_FEATURES = 1000
K_RANGE = range(3, 21)


# Слова, которые встречаются в одном документе или почти во всех, темы не образуют
def build_corpus(token_lists, no_below=2, no_above=0.5):
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
    return dictionary, corpus


def train_lda(corpus, dictionary, topics, passes=LDA_PASSES):
    return LdaModel(corpus, num_topics=topics, id2word=dictionary,
                    passes=passes, random_state=RANDOM_STATE)


def train_nmf(token_lists, topics, max_features=MAX_FEATURES):
    texts = [" ".join(tokens) for tokens in token_lists]
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(texts)

    model = NMF(n_components=topics, random_state=RANDOM_STATE, init="nndsvda", max_iter=1000)
    weights = model.fit_transform(matrix)
    return model, vectorizer, weights


# Топ-слова с весами: для LDA это вероятности слова в теме, для NMF — веса матрицы H
def lda_topics(model, top_words=10):
    return [[(word, float(weight)) for word, weight in model.show_topic(index, top_words)]
            for index in range(model.num_topics)]


def nmf_topics(model, vectorizer, top_words=10):
    names = vectorizer.get_feature_names_out()
    topics = []
    for weights in model.components_:
        top = weights.argsort()[:-top_words - 1:-1]
        topics.append([(names[index], float(weights[index])) for index in top])
    return topics


def coherence_score(topics, token_lists, dictionary):
    words = [[word for word, _ in topic] for topic in topics]
    model = CoherenceModel(topics=words, texts=token_lists, dictionary=dictionary,
                           coherence="c_v")
    return float(model.get_coherence())


# Доля каждой темы в документе; строки нормированы, сумма по теме равна единице
def lda_document_topics(model, corpus, topics):
    matrix = np.zeros((len(corpus), topics))
    for row, bow in enumerate(corpus):
        for topic, share in model.get_document_topics(bow, minimum_probability=0.0):
            matrix[row, topic] = share
    return matrix


def nmf_document_topics(weights):
    totals = weights.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return weights / totals


# Перебор числа тем: для каждого K считаем coherence и берём максимум
def select_topics(method, token_lists, dictionary, corpus, k_range=K_RANGE, passes=LDA_PASSES):
    scores = {}
    for topics in k_range:
        if method == "lda":
            model = train_lda(corpus, dictionary, topics, passes=passes)
            found = lda_topics(model)
        else:
            model, vectorizer, _ = train_nmf(token_lists, topics)
            found = nmf_topics(model, vectorizer)
        scores[topics] = coherence_score(found, token_lists, dictionary)

    best = max(scores, key=scores.get)
    return best, scores
