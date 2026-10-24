"""Распознавание объектов на кадрах."""
import json
from collections import Counter
from pathlib import Path

# YOLO nano: самая лёгкая модель семейства. Выбрана потому, что кадров много,
# а задача — найти крупные объекты, а не отличить породы собак.
MODEL_NAME = "yolov8n.pt"

# Порог уверенности. 0.25 по умолчанию у библиотеки даёт много ложных срабатываний
# на размытых кадрах видео, 0.5 теряет объекты на заднем плане. 0.35 — середина;
# распределение уверенностей пишется в отчёт, чтобы порог можно было проверить.
CONFIDENCE = 0.35

BATCH = 16


def load_model(name=MODEL_NAME):
    from ultralytics import YOLO
    return YOLO(name)


def detect_frames(model, frames, confidence=CONFIDENCE, device=None):
    detections = []
    for start in range(0, len(frames), BATCH):
        batch = frames[start:start + BATCH]
        results = model.predict([item["path"] for item in batch],
                                conf=confidence, verbose=False, device=device)

        for frame, result in zip(batch, results):
            for box in result.boxes:
                detections.append({
                    "time": frame["time"],
                    "label": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                })
    return detections


# Статистика по распознанным сущностям — один из пяти выходов задания
def summarize(detections):
    counts = Counter(item["label"] for item in detections)
    by_label = {}
    for label in counts:
        values = [item["confidence"] for item in detections if item["label"] == label]
        by_label[label] = {
            "detections": len(values),
            "mean_confidence": round(sum(values) / len(values), 3),
            "min_confidence": round(min(values), 3),
            "max_confidence": round(max(values), 3),
        }
    return {
        "total": len(detections),
        "labels": len(counts),
        "by_label": dict(sorted(by_label.items(),
                                key=lambda item: -item[1]["detections"])),
    }


def analyse(decomposed, root, model=None, confidence=CONFIDENCE, force=False, device=None):
    path = Path(root) / decomposed["id"] / "detections.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    model = model or load_model()
    detections = detect_frames(model, decomposed["frames"], confidence, device)
    result = {"id": decomposed["id"], "confidence": confidence,
              "detections": detections, "stats": summarize(detections)}

    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    top = list(result["stats"]["by_label"])[:5]
    print(f"  {decomposed['id']}: детекций {len(detections)}, "
          f"классов {result['stats']['labels']}, чаще всего: {', '.join(top) or 'ничего'}")
    return result
