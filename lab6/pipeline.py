"""Прогон анализа видео от сбора до индекса для интерфейса.

    python pipeline.py --sources videos.txt --output index.json
    python pipeline.py --sources videos.txt --clusters 4 --whisper small --force
"""
import argparse
import json
import sys
from pathlib import Path

import collect
import decompose
import events as events_module
import speech
import topics as topics_module
import vision

ROOT = Path(__file__).resolve().parent / "data"
INDEX = Path(__file__).resolve().parent / "index.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Сбор и семантический анализ видеоданных.")
    parser.add_argument("--sources", default="videos.txt",
                        help="файл со ссылками и путями к локальным роликам")
    parser.add_argument("--output", default=str(INDEX), help="файл индекса")
    parser.add_argument("--data", default=str(ROOT), help="папка промежуточных результатов")
    parser.add_argument("--clusters", type=int, default=5, help="число тематических групп")
    parser.add_argument("--confidence", type=float, default=vision.CONFIDENCE,
                        help="порог уверенности детектора")
    parser.add_argument("--whisper", default=speech.MODEL_SIZE,
                        help="размер модели распознавания речи")
    parser.add_argument("--language", default="ru", help="язык речи в роликах")
    parser.add_argument("--device", default="auto", help="auto, cpu или cuda")
    parser.add_argument("--no-speech", action="store_true", help="пропустить распознавание речи")
    parser.add_argument("--force", action="store_true", help="пересчитать, игнорируя кэш")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    skipped = []

    print("1. Сбор роликов")
    sources = collect.read_sources(args.sources)
    if not sources:
        print(f"  в {args.sources} нет ни одной ссылки")
        return 1
    collected = collect.collect(sources, root, skipped)
    if not collected:
        print("  не собрано ни одного ролика")
        return 1

    print("\n2. Декомпозиция")
    decomposed = [decompose.decompose(meta, root, args.force) for meta in collected]

    print("\n3. Распознавание объектов")
    detector = vision.load_model()
    detections = [vision.analyse(item, root, detector, args.confidence,
                                 args.force, args.device if args.device != "auto" else None)
                  for item in decomposed]

    print("\n4. Речь и громкость")
    recognizer = None if args.no_speech else speech.load_model(args.whisper, device=args.device)
    transcripts = [speech.analyse(item, root, recognizer, args.language, args.force,
                                  recognize=not args.no_speech)
                   for item in decomposed]

    print("\n5. События")
    timelines = [events_module.build(item, found, said, root, args.force)
                 for item, found, said in zip(decomposed, detections, transcripts)]

    print("\n6. Тематическая группировка")
    documents = [{"id": meta["id"], "title": meta["title"],
                  "text": topics_module.document_text(meta, said, found)}
                 for meta, said, found in zip(collected, transcripts, detections)]
    grouped = topics_module.group(documents, root, args.clusters, args.force)
    group_by_id = {row["id"]: row for row in grouped["videos"]}

    index = {
        "videos": [
            {
                "id": meta["id"],
                "title": meta["title"],
                "url": meta.get("webpage_url"),
                "source": meta.get("source"),
                "file": meta["file"],
                "duration": item["duration"],
                "fps": item["fps"],
                "frames": len(item["frames"]),
                "scenes": len(item.get("scenes", [])),
                "cluster": group_by_id[meta["id"]]["cluster"],
                "topic": group_by_id[meta["id"]]["topic"],
                "objects": found["stats"],
                "words": said.get("words", 0),
                "warning": said.get("warning"),
                "segments": said.get("segments", []),
                "events": timeline["events"],
                "event_counts": timeline["counts"],
            }
            for meta, item, found, said, timeline
            in zip(collected, decomposed, detections, transcripts, timelines)
        ],
        "topics": grouped["topics"],
        "cluster_words": grouped["cluster_words"],
        "silhouette": grouped["silhouette"],
        "skipped": skipped,
        "settings": {"confidence": args.confidence, "whisper": args.whisper,
                     "language": args.language, "clusters": grouped["clusters"]},
    }

    Path(args.output).write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    print(f"\nГотово: {args.output}")
    print(f"  роликов {len(index['videos'])}, пропущено {len(skipped)}")
    empty = [video["id"] for video in index["videos"] if not video["words"]]
    if empty:
        print(f"  без транскрипта: {', '.join(empty)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
