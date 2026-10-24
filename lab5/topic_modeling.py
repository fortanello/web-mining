"""Тематическое моделирование веб-данных: LDA и NMF с подбором числа тем.

Примеры:
    python topic_modeling.py --dataset news --method lda --topics auto --output news_topics
    python topic_modeling.py --dataset products --method nmf --topics 15 --output product_categories
    python topic_modeling.py --dataset mixed --method both --topics 10 --compare --output mixed
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import models
import preprocess
import visualize

DATA_DIR = Path(__file__).resolve().parent.parent / "lab4" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="topic_modeling",
        description="Тематическое моделирование датасетов, собранных в практике 2.")
    parser.add_argument("--dataset", required=True, choices=["news", "products", "mixed"],
                        help="какой корпус разбирать")
    parser.add_argument("--method", default="lda", choices=["lda", "nmf", "both"],
                        help="алгоритм: lda, nmf или оба")
    parser.add_argument("--topics", default="auto",
                        help="число тем или auto — подбор по coherence в диапазоне 3..20")
    parser.add_argument("--compare", action="store_true",
                        help="сравнить LDA и NMF на одном датасете")
    parser.add_argument("--output", required=True,
                        help="префикс файлов результата")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="папка с выгрузками практики 2")
    parser.add_argument("--top-words", type=int, default=10,
                        help="сколько слов показывать в теме")
    parser.add_argument("--no-wordcloud", action="store_true",
                        help="не рисовать облака слов")
    return parser.parse_args(argv)


# Одна модель: подбор K при необходимости, обучение, темы, coherence, доли тем
def run_method(method, token_lists, dictionary, corpus, topics_option, top_words):
    scores = None
    if topics_option == "auto":
        topics, scores = models.select_topics(method, token_lists, dictionary, corpus)
        print(f"  {method.upper()}: подобрано K = {topics} (c_v = {scores[topics]:.4f})")
    else:
        topics = int(topics_option)

    if method == "lda":
        model = models.train_lda(corpus, dictionary, topics)
        found = models.lda_topics(model, top_words)
        shares = models.lda_document_topics(model, corpus, topics)
    else:
        model, vectorizer, weights = models.train_nmf(token_lists, topics)
        found = models.nmf_topics(model, vectorizer, top_words)
        shares = models.nmf_document_topics(weights)

    coherence = models.coherence_score(found, token_lists, dictionary)
    print(f"  {method.upper()}: тем {topics}, coherence c_v = {coherence:.4f}")

    return {"method": method, "topics": topics, "coherence": coherence,
            "words": found, "shares": shares, "scores": scores}


def save_topics_json(path, dataset, stats, results):
    payload = {
        "dataset": dataset,
        "documents": stats,
        "results": [
            {
                "method": result["method"],
                "topics": result["topics"],
                "coherence": result["coherence"],
                "coherence_by_k": result["scores"],
                "keywords": [
                    {"topic": index,
                     "words": [{"word": word, "weight": weight} for word, weight in topic]}
                    for index, topic in enumerate(result["words"])
                ],
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# Для каждого документа — главная тема и доли всех тем
def save_documents_csv(path, documents, result):
    shares = result["shares"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["url", "title", "source", "главная тема", "доля главной темы"]
                        + [f"тема {index}" for index in range(result["topics"])])

        for row, document in enumerate(documents):
            weights = shares[row]
            leader = int(weights.argmax())
            writer.writerow([document["url"], document["title"], document["source"],
                             leader, round(float(weights[leader]), 4)]
                            + [round(float(value), 4) for value in weights])


def main(argv=None):
    args = parse_args(argv)
    OUTPUT_DIR.mkdir(exist_ok=True)

    methods = ["lda", "nmf"] if (args.method == "both" or args.compare) else [args.method]

    print(f"Датасет: {args.dataset}")
    documents, stats = preprocess.prepare(args.dataset, args.data_dir)
    print(f"  документов: загружено {stats['loaded']}, "
          f"отброшено коротких {stats['dropped']}, в работе {stats['kept']}")
    if stats["kept"] < 20:
        print("  предупреждение: документов слишком мало, темы будут неустойчивы")

    token_lists = [document["tokens"] for document in documents]
    dictionary, corpus = models.build_corpus(token_lists)
    print(f"  словарь после фильтрации: {len(dictionary)} слов")

    results = [run_method(method, token_lists, dictionary, corpus,
                          args.topics, args.top_words)
               for method in methods]

    prefix = OUTPUT_DIR / args.output
    save_topics_json(Path(f"{prefix}_topics.json"), args.dataset, stats, results)

    for result in results:
        name = result["method"]
        save_documents_csv(Path(f"{prefix}_documents_{name}.csv"), documents, result)

        visualize.plot_top_words(
            result["words"], f"{name.upper()}, темы корпуса «{args.dataset}»",
            f"{prefix}_topwords_{name}.png")
        visualize.plot_document_topics(
            result["shares"], f"{prefix}_documents_{name}.png",
            title=f"{name.upper()}: распределение тем по документам")

        if result["scores"]:
            visualize.plot_coherence(result["scores"], result["topics"],
                                     f"{prefix}_coherence_{name}.png",
                                     title=f"{name.upper()}: подбор числа тем по coherence")

        if not args.no_wordcloud:
            visualize.plot_wordclouds(result["words"], f"{prefix}_wordcloud_{name}.png")

    if len(results) == 2:
        visualize.plot_comparison(results[0]["words"], results[1]["words"],
                                  f"{prefix}_compare.png")
        better = max(results, key=lambda item: item["coherence"])
        print(f"  сравнение: выше coherence у {better['method'].upper()} "
              f"({better['coherence']:.4f})")

    print(f"Результаты: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.glob(f"{args.output}_*")):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
