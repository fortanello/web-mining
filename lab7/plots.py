import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


NAMES = {"popular": "Популярное", "content": "TF-IDF", "svd": "SVD", "als": "ALS item-based"}
COLORS = {"popular": "#60666d", "content": "#2475a5", "svd": "#aa6228", "als": "#17775d"}


def save(figure, output, name):
    figure.tight_layout()
    figure.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_results(ratings, results, summary, output):
    reference_k = summary["reference_k"]
    overall = results[results["segment"] == "все"]
    reference = overall[overall["k"] == reference_k]

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    counts = ratings["rating"].value_counts().sort_index()
    axes[0].bar(counts.index, counts, width=0.42, color="#2475a5")
    axes[0].set(title="Оценки фильмов", xlabel="Оценка", ylabel="Число оценок",
                xticks=[1, 2, 3, 4, 5])
    activity = ratings.groupby("userId").size()
    bins = np.geomspace(activity.min(), activity.max(), 18)
    axes[1].hist(activity, bins=bins, color="#2475a5")
    axes[1].set(title="Активность", xlabel="Оценок у пользователя (лог. шкала)",
                ylabel="Число пользователей", xscale="log")
    filled = 1 - summary["sparsity"]
    axes[2].barh(["Матрица оценок"], [filled], color="#2475a5", label="Есть оценка")
    axes[2].barh(["Матрица оценок"], [1 - filled], left=[filled], color="#e2e8ee", label="Пусто")
    axes[2].set(title=f"Заполнено {filled:.2%}", xlabel="Доля ячеек", xlim=(0, 1))
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(loc="upper right")
    figure.savefig(output / "data.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for model, group in overall.groupby("model", sort=False):
        for axis, metric in zip(axes, ["ndcg", "recall"]):
            axis.plot(group["k"], group[metric], marker="o", color=COLORS[model], label=NAMES[model])
    for axis, name in zip(axes, ["NDCG@K", "Recall@K"]):
        axis.set(xlabel="Длина выдачи K", ylabel=name, xticks=sorted(overall["k"].unique()), ylim=(0, None))
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    save(figure, output, "ranking.png")

    by_segment = results[(results["k"] == reference_k) & (results["segment"] != "все")]
    if not by_segment.empty:
        pivot = by_segment.pivot(index="segment", columns="model", values="ndcg")
        pivot = pivot.reindex([label for label in summary["segments"] if label in pivot.index])
        models = reference["model"].tolist()
        pivot = pivot[models].rename(columns=NAMES)
        pivot.index = [f"{label}\nn = {summary['segments'][label]}" for label in pivot.index]
        axis = pivot.plot.bar(figsize=(8, 3.5), color=[COLORS[model] for model in models], rot=0)
        axis.set(title=f"Качество по длине истории", ylabel=f"NDCG@{reference_k}",
                 xlabel="Число оценок в обучающей части")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        save(axis.figure, output, "segments.png")

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    names = [NAMES[model] for model in reference["model"]]
    colors = [COLORS[model] for model in reference["model"]]
    for axis, column, title in zip(axes, ["coverage", "diversity"],
                                  ["Покрытие каталога", "Жанровое разнообразие"]):
        bars = axis.barh(names, reference[column], color=colors)
        digits = 2 if column == "coverage" else 1
        axis.bar_label(bars, labels=[f"{value:.{digits}%}" for value in reference[column]], padding=3)
        axis.invert_yaxis()
        upper = reference[column].max() * 1.3 if column == "coverage" else 1.0
        axis.set(title=f"{title} при K = {reference_k}", xlabel="Доля", xlim=(0, upper))
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=1 if column == "coverage" else 0))
    save(figure, output, "catalog.png")
