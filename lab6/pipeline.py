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


# Видеокарта новее, чем сборка torch, — частый случай: карта есть, CUDA видна,
# но ядер под её архитектуру в сборке нет, и расчёт падает уже внутри модели.
# Проверяем заранее и честно откатываемся на процессор
def check_device(device):
    try:
        import torch
    except ImportError:
        return "cpu"

    if device == "cpu" or not torch.cuda.is_available():
        if device == "cuda":
            print("  CUDA недоступна, считаю на процессоре")
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()

    if architecture not in supported:
        print(f"  ВНИМАНИЕ: {torch.cuda.get_device_name(0)} имеет архитектуру "
              f"{architecture}, а torch собран под {', '.join(supported)}")
        print("  переустановите torch под свежую CUDA, иначе видеокарта бесполезна:")
        print("      pip install torch torchvision --force-reinstall \\")
        print("          --index-url https://download.pytorch.org/whl/cu130")
        print("  пока считаю на процессоре")
        return "cpu"

    print(f"  видеокарта: {torch.cuda.get_device_name(0)} ({architecture})")
    return "cuda"


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

    print("0. Устройство расчёта")
    device = check_device(args.device)

    print("1. Сбор роликов")
    sources = collect.read_sources(args.sources)
    if not sources:
        print(f"  в {args.sources} нет ни одной ссылки")
        return 1
    collected = collect.collect(sources, root, skipped, args.force)
    if not collected:
        print("  не собрано ни одного ролика")
        return 1

    print("\n2. Декомпозиция")
    decomposed = [decompose.decompose(meta, root, args.force) for meta in collected]

    print("\n3. Распознавание объектов")
    detector = vision.load_model()
    detections = [vision.analyse(item, root, detector, args.confidence, args.force, device)
                  for item in decomposed]

    print("\n4. Речь и громкость")
    recognizer = None if args.no_speech else speech.load_model(args.whisper, device=device)
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
                     "language": args.language, "clusters": grouped["clusters"],
                     "device": device},
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
