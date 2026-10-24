"""Декомпозиция ролика: аудио для распознавания, кадры с метками, границы сцен."""
import json
import re
import subprocess
from pathlib import Path

# Whisper ждёт моно 16 кГц — пересчитываем сразу, чтобы не делать это в памяти
AUDIO_RATE = 16000

# Сколько кадров брать с ролика. Одинаковая частота для всех роликов — ошибка:
# на пятиминутном это 300 кадров, на часовом — 3600, и анализ встаёт.
# Поэтому частота считается от длительности, с ограничением сверху и снизу.
TARGET_FRAMES = 240
MIN_FPS = 0.1
MAX_FPS = 1.0

# Порог смены сцены у ffmpeg: доля изменившихся пикселей
SCENE_THRESHOLD = 0.4


def frame_rate(duration):
    if not duration or duration <= 0:
        return MAX_FPS
    return round(min(MAX_FPS, max(MIN_FPS, TARGET_FRAMES / duration)), 3)


def extract_audio(video_path, audio_path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn",
         "-ac", "1", "-ar", str(AUDIO_RATE), str(audio_path)],
        check=True)
    return audio_path


# Кадры нумеруются по порядку, метка времени восстанавливается из номера и частоты
def extract_frames(video_path, frames_dir, fps):
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-vf", f"fps={fps}", "-q:v", "3", str(frames_dir / "%05d.jpg")],
        check=True)

    frames = sorted(frames_dir.glob("*.jpg"))
    return [{"path": str(path), "time": round(index / fps, 2)}
            for index, path in enumerate(frames)]


# Границы сцен даёт сам ffmpeg: кадр считается новой сценой, если картинка
# изменилась больше порога. Это дешевле, чем сравнивать кадры в Python
def detect_scenes(video_path, threshold=SCENE_THRESHOLD):
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(video_path),
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True)

    times = [round(float(match), 2)
             for match in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    return sorted(set(times))


def decompose(meta, root, force=False):
    folder = Path(root) / meta["id"]
    audio_path = folder / "audio.wav"
    frames_dir = folder / "frames"
    index_path = folder / "decomposed.json"

    # Промежуточные результаты хранятся в файлах: повторный прогон тяжёлых
    # моделей занимает часы, и второй раз делать ту же работу незачем
    if index_path.exists() and not force:
        return json.loads(index_path.read_text(encoding="utf-8"))

    duration = meta["probe"]["duration"]
    fps = frame_rate(duration)

    if meta["probe"]["has_audio"]:
        extract_audio(meta["file"], audio_path)
    else:
        print(f"  {meta['id']}: в ролике нет звуковой дорожки")

    frames = extract_frames(meta["file"], frames_dir, fps)
    scenes = detect_scenes(meta["file"])

    result = {
        "id": meta["id"],
        "fps": fps,
        "duration": duration,
        "audio": str(audio_path) if meta["probe"]["has_audio"] else None,
        "frames": frames,
        "scenes": scenes,
    }
    index_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {meta['id']}: кадров {len(frames)} при {fps} к/с, сцен {len(scenes)}")
    return result
