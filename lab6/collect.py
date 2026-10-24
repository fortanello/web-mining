"""Сбор роликов: метаданные с хостинга, файл, технические параметры."""
import json
import subprocess
import time
from pathlib import Path

import yt_dlp

# Пауза между загрузками: массовый сбор без неё нарушает условия площадок
DOWNLOAD_PAUSE = 2.0

META_FIELDS = ("id", "title", "description", "duration", "uploader", "upload_date",
               "view_count", "like_count", "categories", "tags", "webpage_url")


def video_dir(root, video_id):
    path = Path(root) / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# Технические параметры файла: ffprobe знает то, чего нет в метаданных хостинга
def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), {})

    rate = video.get("avg_frame_rate", "0/1")
    numerator, _, denominator = rate.partition("/")
    fps = float(numerator) / float(denominator) if denominator and float(denominator) else 0.0

    return {
        "duration": float(data["format"].get("duration", 0.0)),
        "size_bytes": int(data["format"].get("size", 0)),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": round(fps, 3),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "has_audio": bool(audio),
    }


def download(url, root, max_height=720):
    options = {
        "outtmpl": str(Path(root) / "%(id)s" / "source.%(ext)s"),
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)

    meta = {field: info.get(field) for field in META_FIELDS}
    meta["source"] = "hosting"
    folder = video_dir(root, info["id"])
    files = [item for item in folder.iterdir() if item.name.startswith("source")]
    if not files:
        raise FileNotFoundError(f"файл не скачался: {url}")

    meta["file"] = str(files[0])
    meta["probe"] = probe(files[0])
    return meta


# Локальный файл: метаданных хостинга нет, берём имя и показания ffprobe
def take_local(path, root):
    path = Path(path)
    video_id = "local_" + path.stem.replace(" ", "_")[:40]
    folder = video_dir(root, video_id)
    target = folder / f"source{path.suffix}"
    if not target.exists():
        target.write_bytes(path.read_bytes())

    return {
        "id": video_id,
        "title": path.stem,
        "description": "",
        "duration": None,
        "uploader": None,
        "upload_date": None,
        "view_count": None,
        "like_count": None,
        "categories": [],
        "tags": [],
        "webpage_url": None,
        "source": "local",
        "file": str(target),
        "probe": probe(target),
    }


def read_sources(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# Ролик, который не удалось взять, не исчезает молча: причина пишется в отчёт
def collect(sources, root, skipped):
    collected = []
    for source in sources:
        try:
            if source.startswith("http"):
                meta = download(source, root)
                time.sleep(DOWNLOAD_PAUSE)
            else:
                meta = take_local(source, root)
        except Exception as error:
            skipped.append({"source": source, "reason": f"{type(error).__name__}: {error}"})
            print(f"  пропущен {source}: {type(error).__name__}")
            continue

        meta_path = video_dir(root, meta["id"]) / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        collected.append(meta)
        print(f"  {meta['id']}: {str(meta['title'])[:60]} "
              f"({meta['probe']['duration']:.0f} с)")

    return collected
