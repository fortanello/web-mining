"""Аудиодорожка: громкость по окнам и транскрипт речи."""
import json
import os
import sys
import types
import wave
from pathlib import Path

import numpy as np

# base — компромисс: small заметно точнее на русском, но втрое медленнее,
# tiny путает термины. На видеокарте можно безболезненно брать small.
MODEL_SIZE = "base"
WINDOW = 1.0


# Отсчёты в диапазоне [-1, 1] — в таком виде их ждёт и расчёт громкости,
# и модель распознавания. К 16 кГц моно дорожку привёл ffmpeg при декомпозиции
def read_samples(audio_path):
    with wave.open(str(audio_path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, rate


# ctranslate2 внутри faster-whisper собран под CUDA 12 и ищет её библиотеки
# по имени файла, а torch приносит с собой CUDA 13. Библиотеки CUDA 12 ставятся
# пакетами nvidia-cublas-cu12 и nvidia-cudnn-cu12, но путь к ним Windows сама
# не знает — подсказываем до загрузки модели
def add_cuda_libraries():
    try:
        import nvidia
    except ImportError:
        return

    folders = sorted({str(dll.parent) for root in nvidia.__path__
                      for dll in Path(root).rglob("*.dll")})
    for folder in folders:
        os.add_dll_directory(folder)
    if folders:
        os.environ["PATH"] = os.pathsep.join(folders + [os.environ["PATH"]])


# faster-whisper тянет PyAV только чтобы прочитать файл с диска, а мы отдаём
# готовые отсчёты. Windows блокирует неподписанные библиотеки PyAV политикой
# управления приложениями, поэтому ненужную зависимость подменяем пустышкой
def load_model(size=MODEL_SIZE, device="auto", compute_type="default"):
    try:
        import av
    except ImportError as error:
        print(f"  PyAV не загрузился ({error}), читаю дорожку сам — "
              f"на распознавание это не влияет")
        sys.modules["av"] = types.ModuleType("av")

    add_cuda_libraries()
    from faster_whisper import WhisperModel
    return WhisperModel(size, device=device, compute_type=compute_type)


# Громкость считается по окнам в один корень из среднего квадрата:
# по ней видно всплески, которые часто совпадают со значимыми моментами
def loudness(audio_path, window=WINDOW):
    samples, rate = read_samples(audio_path)
    if samples.size == 0:
        return []

    step = int(rate * window)
    windows = []
    for start in range(0, samples.size, step):
        chunk = samples[start:start + step]
        if chunk.size == 0:
            continue
        windows.append({"time": round(start / rate, 2),
                        "rms": round(float(np.sqrt(np.mean(chunk ** 2))), 5)})
    return windows


def transcribe(model, audio_path, language="ru"):
    samples, _ = read_samples(audio_path)
    segments, info = model.transcribe(samples, language=language, vad_filter=True)
    return [{"start": round(segment.start, 2), "end": round(segment.end, 2),
             "text": segment.text.strip()} for segment in segments], info


def analyse(decomposed, root, model=None, language="ru", force=False, recognize=True):
    path = Path(root) / decomposed["id"] / "transcript.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    audio = decomposed.get("audio")
    if not audio or not Path(audio).exists():
        result = {"id": decomposed["id"], "segments": [], "loudness": [],
                  "warning": "нет звуковой дорожки"}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {decomposed['id']}: звука нет, транскрипт пустой")
        return result

    # Громкость не зависит от модели распознавания, поэтому считается всегда
    windows = loudness(audio)

    if recognize:
        model = model or load_model()
        segments, _ = transcribe(model, audio, language)
    else:
        segments = []

    words = sum(len(segment["text"].split()) for segment in segments)
    result = {"id": decomposed["id"], "language": language,
              "segments": segments, "loudness": windows, "words": words}
    if not recognize:
        result["warning"] = "распознавание речи отключено ключом --no-speech"

    # Пустой транскрипт легко не заметить, а он означает, что анализ речи
    # не состоялся: ролик без слов, чужой язык или сломанная дорожка
    if words == 0 and recognize:
        result["warning"] = "транскрипт пуст: проверьте язык и звуковую дорожку"
        print(f"  {decomposed['id']}: ВНИМАНИЕ, транскрипт пуст")
    else:
        print(f"  {decomposed['id']}: сегментов {len(segments)}, слов {words}")

    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
