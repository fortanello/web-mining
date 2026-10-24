"""Интерфейс: поиск по роликам и переход на нужную секунду.

    uvicorn api:app --reload
    затем открыть http://127.0.0.1:8000
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
INDEX = BASE / "index.json"
STATIC = BASE / "static"

app = FastAPI(title="Анализ видеоданных", version="1.0")


def load_index():
    if not INDEX.exists():
        raise HTTPException(status_code=503,
                            detail="нет index.json — сначала запустите pipeline.py")
    return json.loads(INDEX.read_text(encoding="utf-8"))


# Ссылка с временной меткой: YouTube принимает параметр t и открывает нужную секунду
def moment_url(video, second):
    url = video.get("url")
    if not url:
        return None
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={int(second)}s"


@app.get("/api/videos")
def videos():
    index = load_index()
    return {
        "videos": [
            {key: video[key] for key in
             ("id", "title", "url", "duration", "cluster", "topic", "words", "event_counts")}
            for video in index["videos"]
        ],
        "cluster_words": index["cluster_words"],
        "topics": index["topics"],
        "silhouette": index["silhouette"],
        "skipped": index["skipped"],
        "settings": index["settings"],
    }


@app.get("/api/video/{video_id}")
def video(video_id: str):
    index = load_index()
    found = next((item for item in index["videos"] if item["id"] == video_id), None)
    if found is None:
        raise HTTPException(status_code=404, detail=f"нет ролика {video_id}")
    return found


# Поиск идёт и по словам речи, и по названиям найденных объектов:
# на запрос «кот» должны найтись и произнесённое слово, и кадры с животным
@app.get("/api/search")
def search(q: str, limit: int = 100):
    index = load_index()
    needle = q.strip().lower()
    if not needle:
        return {"query": q, "hits": []}

    hits = []
    for video in index["videos"]:
        for segment in video.get("segments", []):
            if needle in segment["text"].lower():
                hits.append({"video": video["id"], "title": video["title"],
                             "kind": "речь", "start": segment["start"],
                             "text": segment["text"],
                             "url": moment_url(video, segment["start"])})

        for event in video.get("events", []):
            if event["type"] == "object" and needle in event["label"].lower():
                hits.append({"video": video["id"], "title": video["title"],
                             "kind": "объект", "start": event["start"],
                             "text": f"{event['label']} в кадре "
                                     f"{event['start']}–{event['end']} с",
                             "url": moment_url(video, event["start"])})

    hits.sort(key=lambda hit: (hit["video"], hit["start"]))
    return {"query": q, "found": len(hits), "hits": hits[:limit]}


@app.get("/api/timeline/{video_id}")
def timeline(video_id: str):
    found = video(video_id)
    return {"id": found["id"], "duration": found["duration"],
            "events": found["events"], "counts": found["event_counts"]}


@app.get("/")
def index_page():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
