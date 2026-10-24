"""События ролика. Событие — это интервал времени, а не отдельный кадр."""
import json
import statistics
from pathlib import Path

# Разрыв, при котором два появления объекта считаются одним событием.
# Берётся в шагах между кадрами: при 0.5 к/с соседние кадры отстоят на 2 с,
# и требовать непрерывности бессмысленно
GAP_IN_FRAMES = 2.0

# Минимальная длительность события по объекту: одиночное срабатывание
# на одном кадре чаще всего ложное
MIN_DURATION = 0.0

# Во сколько раз громкость должна превысить обычную, чтобы считаться всплеском
LOUD_FACTOR = 1.6


# Детекции одного класса, идущие подряд, склеиваются в интервал
def object_events(detections, fps):
    gap = GAP_IN_FRAMES / max(fps, 0.01)
    by_label = {}
    for item in sorted(detections, key=lambda row: (row["label"], row["time"])):
        by_label.setdefault(item["label"], []).append(item)

    events = []
    for label, items in by_label.items():
        start = previous = items[0]["time"]
        best = items[0]["confidence"]
        count = 1

        for item in items[1:]:
            if item["time"] - previous <= gap:
                previous = item["time"]
                best = max(best, item["confidence"])
                count += 1
                continue

            events.append({"type": "object", "label": label, "start": start,
                           "end": previous, "confidence": best, "frames": count})
            start = previous = item["time"]
            best = item["confidence"]
            count = 1

        events.append({"type": "object", "label": label, "start": start,
                       "end": previous, "confidence": best, "frames": count})

    return [event for event in events if event["end"] - event["start"] >= MIN_DURATION]


# Всплеск громкости: окна, заметно превышающие медиану по ролику
def loud_events(windows, factor=LOUD_FACTOR):
    values = [window["rms"] for window in windows if window["rms"] > 0]
    if len(values) < 5:
        return []

    threshold = statistics.median(values) * factor
    events, current = [], None

    for window in windows:
        if window["rms"] >= threshold:
            if current is None:
                current = {"type": "loud", "label": "громкий момент",
                           "start": window["time"], "end": window["time"],
                           "rms": window["rms"]}
            else:
                current["end"] = window["time"]
                current["rms"] = max(current["rms"], window["rms"])
        elif current is not None:
            events.append(current)
            current = None

    if current is not None:
        events.append(current)
    return events


def speech_events(segments):
    return [{"type": "speech", "label": segment["text"][:80],
             "start": segment["start"], "end": segment["end"],
             "text": segment["text"]}
            for segment in segments if segment["text"]]


def scene_events(scenes, duration):
    events = []
    for index, start in enumerate(scenes):
        end = scenes[index + 1] if index + 1 < len(scenes) else duration
        events.append({"type": "scene", "label": f"сцена {index + 1}",
                       "start": start, "end": round(end or start, 2)})
    return events


def build(decomposed, detections, transcript, root, force=False):
    path = Path(root) / decomposed["id"] / "events.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    events = (object_events(detections["detections"], decomposed["fps"])
              + loud_events(transcript.get("loudness", []))
              + speech_events(transcript.get("segments", []))
              + scene_events(decomposed.get("scenes", []), decomposed.get("duration")))
    events.sort(key=lambda event: (event["start"], event["type"]))

    counts = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1

    result = {"id": decomposed["id"], "duration": decomposed.get("duration"),
              "events": events, "counts": counts}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {decomposed['id']}: событий {len(events)} ({counts})")
    return result
