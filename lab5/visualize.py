"""Графики по результатам тематического моделирования."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150


def _grid(count, columns=5):
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, min(count, columns),
                                figsize=(min(count, columns) * 4.2, rows * 3.6),
                                squeeze=False)
    return figure, axes.flatten()


# Топ-слова каждой темы горизонтальными столбиками: самое весомое слово сверху
def plot_top_words(topics, title, path, columns=5):
    figure, axes = _grid(len(topics), columns)

    for index, topic in enumerate(topics):
        words = [word for word, _ in topic][::-1]
        weights = [weight for _, weight in topic][::-1]

        axis = axes[index]
        axis.barh(words, weights, color="#2b7bba")
        axis.set_title(f"Тема {index}", fontsize=11)
        axis.tick_params(labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    for axis in axes[len(topics):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


# Зависимость coherence от числа тем с отметкой выбранного K
def plot_coherence(scores, best, path, title="Подбор числа тем по coherence (c_v)"):
    keys = sorted(scores)
    values = [scores[key] for key in keys]

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(keys, values, marker="o", color="#2b7bba")
    axis.axvline(best, color="#c0392b", linestyle="--",
                 label=f"выбрано K = {best} (c_v = {scores[best]:.3f})")
    axis.set_xlabel("Число тем K")
    axis.set_ylabel("Coherence c_v")
    axis.set_title(title)
    axis.set_xticks(keys)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


# Тепловая карта «документ — тема» по первым документам плюс общий вес тем
def plot_document_topics(matrix, path, limit=40, title="Распределение тем по документам"):
    shown = matrix[:limit]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2, 1]})

    image = left.imshow(shown, aspect="auto", cmap="viridis")
    left.set_xlabel("Тема")
    left.set_ylabel("Документ")
    left.set_title(f"Первые {len(shown)} документов")
    left.set_xticks(range(matrix.shape[1]))
    figure.colorbar(image, ax=left, label="Доля темы")

    totals = matrix.sum(axis=0)
    right.bar(range(len(totals)), totals, color="#2b7bba")
    right.set_xlabel("Тема")
    right.set_ylabel("Суммарный вес по корпусу")
    right.set_title("Насколько тема представлена")
    right.set_xticks(range(len(totals)))
    right.grid(alpha=0.3, axis="y")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


# Облака слов: размер слова пропорционален его весу в теме
def plot_wordclouds(topics, path, columns=4):
    from wordcloud import WordCloud

    figure, axes = _grid(len(topics), columns)
    for index, topic in enumerate(topics):
        weights = {word: weight for word, weight in topic}
        cloud = WordCloud(width=600, height=400, background_color="white",
                          colormap="viridis").generate_from_frequencies(weights)
        axes[index].imshow(cloud, interpolation="bilinear")
        axes[index].set_title(f"Тема {index}", fontsize=11)
        axes[index].axis("off")

    for axis in axes[len(topics):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)


# Темы двух методов рядом — так видно, какой даёт более резкие темы
def plot_comparison(lda, nmf, path, top_words=8):
    rows = max(len(lda), len(nmf))
    figure, axes = plt.subplots(rows, 2, figsize=(11, rows * 1.5), squeeze=False)

    for column, (topics, name) in enumerate(((lda, "LDA"), (nmf, "NMF"))):
        for row in range(rows):
            axis = axes[row][column]
            axis.axis("off")
            if row >= len(topics):
                continue
            words = ", ".join(word for word, _ in topics[row][:top_words])
            axis.text(0, 0.5, f"{row}:  {words}", fontsize=10, va="center")
            if row == 0:
                axis.set_title(name, fontsize=13, loc="left")

    figure.suptitle("Сравнение тем: LDA и NMF", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
