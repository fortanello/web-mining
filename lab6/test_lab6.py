import json
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
import collect
import decompose
import events
import pipeline
import speech
import topics
import vision

# ---- декомпозиция ----

def test_frame_rate_adapts_to_duration():
    # одинаковая частота для всех роликов — типичная ошибка: на часовом видео
    # это тысячи кадров, поэтому частота падает с ростом длительности
    assert decompose.frame_rate(30) == decompose.MAX_FPS
    assert decompose.frame_rate(3600) == decompose.MIN_FPS
    assert decompose.frame_rate(600) < decompose.frame_rate(60)

def test_frame_rate_survives_unknown_duration():
    assert decompose.frame_rate(None) == decompose.MAX_FPS
    assert decompose.frame_rate(0) == decompose.MAX_FPS

def test_frame_count_stays_bounded():
    for duration in (60, 300, 1800, 7200):
        frames = duration * decompose.frame_rate(duration)
        assert frames <= decompose.TARGET_FRAMES * 3

# ---- события ----

def detections(times, label="person", confidence=0.8):
    return [{"time": float(t), "label": label, "confidence": confidence} for t in times]

def test_consecutive_detections_become_one_interval():
    found = events.object_events(detections([0, 1, 2, 3]), fps=1.0)

    assert len(found) == 1
    assert (found[0]["start"], found[0]["end"]) == (0.0, 3.0)
    assert found[0]["frames"] == 4

def test_gap_splits_into_two_events():
    found = events.object_events(detections([0, 1, 30, 31]), fps=1.0)

    assert len(found) == 2
    assert [(item["start"], item["end"]) for item in found] == [(0.0, 1.0), (30.0, 31.0)]

def test_allowed_gap_scales_with_frame_rate():
    # при 0.2 к/с соседние кадры отстоят на 5 с, разрыв в 10 с ещё не разрыв
    sparse = events.object_events(detections([0, 10]), fps=0.2)
    dense = events.object_events(detections([0, 10]), fps=1.0)

    assert len(sparse) == 1
    assert len(dense) == 2

def test_different_labels_do_not_merge():
    found = events.object_events(detections([0, 1]) + detections([0, 1], label="cat"), fps=1.0)

    assert {item["label"] for item in found} == {"person", "cat"}
    assert len(found) == 2

def test_event_keeps_highest_confidence():
    found = events.object_events(
        detections([0], confidence=0.4) + detections([1], confidence=0.9), fps=1.0)

    assert found[0]["confidence"] == 0.9

def test_loud_events_need_enough_windows():
    assert events.loud_events([{"time": 0.0, "rms": 0.5}]) == []

def test_loud_events_find_the_spike():
    windows = [{"time": float(t), "rms": 0.01} for t in range(10)]
    windows[4]["rms"] = windows[5]["rms"] = 0.5

    found = events.loud_events(windows)
    assert len(found) == 1
    assert (found[0]["start"], found[0]["end"]) == (4.0, 5.0)

def test_scene_events_close_the_last_scene_with_duration():
    found = events.scene_events([0.0, 10.0], duration=25.0)

    assert found[-1]["end"] == 25.0
    assert len(found) == 2

# ---- распознавание ----

def test_summarize_counts_entities():
    stats = vision.summarize(detections([0, 1]) + detections([2], label="cat", confidence=0.5))

    assert stats["total"] == 3
    assert stats["labels"] == 2
    assert stats["by_label"]["person"]["detections"] == 2
    assert stats["by_label"]["cat"]["max_confidence"] == 0.5

def test_labels_sorted_by_frequency():
    stats = vision.summarize(detections([0, 1, 2]) + detections([0], label="cat"))

    assert list(stats["by_label"]) == ["person", "cat"]

# ---- аудио ----

def write_wave(path, samples, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((samples * 32767).astype(np.int16).tobytes())

def test_loudness_follows_the_signal(tmp_path):
    rate = 16000
    quiet = np.zeros(rate, dtype=np.float32)
    loud = np.ones(rate, dtype=np.float32) * 0.5
    path = tmp_path / "audio.wav"
    write_wave(path, np.concatenate([quiet, loud]), rate)

    windows = speech.loudness(path, window=1.0)
    assert len(windows) == 2
    assert windows[0]["rms"] < windows[1]["rms"]

def test_empty_transcript_is_flagged(tmp_path, capsys):
    path = tmp_path / "audio.wav"
    write_wave(path, np.zeros(16000, dtype=np.float32))
    decomposed = {"id": "video", "audio": str(path), "fps": 1.0, "frames": []}
    (tmp_path / "video").mkdir()

    class Silent:
        def transcribe(self, *args, **kwargs):
            return iter([]), None

    result = speech.analyse(decomposed, tmp_path, Silent(), force=True)

    # пустой транскрипт легко не заметить, поэтому он помечается явно
    assert result["words"] == 0
    assert "пуст" in result["warning"]
    assert result["loudness"], "громкость считается даже без распознанной речи"

def test_loudness_computed_when_recognition_disabled(tmp_path):
    path = tmp_path / "audio.wav"
    write_wave(path, np.ones(16000, dtype=np.float32) * 0.3)
    (tmp_path / "video").mkdir()
    decomposed = {"id": "video", "audio": str(path), "fps": 1.0, "frames": []}

    result = speech.analyse(decomposed, tmp_path, None, force=True, recognize=False)
    assert result["loudness"]
    assert "отключено" in result["warning"]

def test_missing_audio_does_not_crash(tmp_path):
    (tmp_path / "video").mkdir()
    result = speech.analyse({"id": "video", "audio": None}, tmp_path, force=True)

    assert result["segments"] == []
    assert "нет звуковой дорожки" in result["warning"]

# ---- темы ----

def test_document_text_uses_speech_and_objects():
    text = topics.document_text(
        {"title": "Ремонт кухни", "description": "плитка", "tags": ["ремонт"]},
        {"segments": [{"text": "кладём плитку"}]},
        {"stats": {"by_label": {"person": {}, "sink": {}}}})

    for word in ("ремонт", "плитка", "кладём", "person", "sink"):
        assert word in text

def test_clustering_skipped_for_few_videos():
    _, matrix = topics.build_matrix(["кот сидит", "ремонт кухни"])
    labels, used, score = topics.cluster(matrix, clusters=5)

    assert used == 1
    assert score == 0.0
    assert set(labels) == {0}

def test_clustering_groups_similar_videos():
    texts = ["ремонт кухни плитка", "ремонт ванной плитка", "ремонт кухни кран",
             "кот играет мячик", "кот спит диван", "кот ест корм"]
    _, matrix = topics.build_matrix(texts)
    labels, used, score = topics.cluster(matrix, clusters=2)

    assert used == 2
    assert len(set(labels[:3])) == 1          # все про ремонт в одной группе
    assert len(set(labels[3:])) == 1          # все про кота в другой
    assert labels[0] != labels[3]
    assert score > 0

# ---- интерфейс ----

INDEX = {
    "videos": [{
        "id": "abc123", "title": "Ремонт кухни", "url": "https://youtu.be/abc123",
        "duration": 60.0, "fps": 1.0, "frames": 60, "scenes": 2, "cluster": 0, "topic": 0,
        "objects": {"total": 2, "labels": 1, "by_label": {"sink": {"detections": 2}}},
        "words": 5, "warning": None,
        "segments": [{"start": 12.0, "end": 15.0, "text": "кладём плитку на стену"}],
        "events": [{"type": "object", "label": "sink", "start": 20.0, "end": 25.0,
                    "confidence": 0.8, "frames": 6}],
        "event_counts": {"object": 1},
    }],
    "topics": [[{"word": "плитка", "weight": 0.5}]],
    "cluster_words": {"0": ["ремонт", "плитка"]},
    "silhouette": 0.0, "skipped": [],
    "settings": {"confidence": 0.35, "whisper": "base", "language": "ru", "clusters": 1},
}

@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "index.json"
    path.write_text(json.dumps(INDEX, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(api, "INDEX", path)
    return TestClient(api.app)

def test_search_finds_spoken_word_with_timestamp(client):
    data = client.get("/api/search", params={"q": "плитку"}).json()

    assert data["found"] == 1
    hit = data["hits"][0]
    assert hit["kind"] == "речь"
    assert hit["start"] == 12.0
    assert hit["url"].endswith("t=12s")

def test_search_finds_object_in_frame(client):
    data = client.get("/api/search", params={"q": "sink"}).json()

    assert data["found"] == 1
    assert data["hits"][0]["kind"] == "объект"
    assert data["hits"][0]["start"] == 20.0

def test_search_is_case_insensitive(client):
    assert client.get("/api/search", params={"q": "ПЛИТКУ"}).json()["found"] == 1

def test_empty_query_returns_nothing(client):
    assert client.get("/api/search", params={"q": "   "}).json()["hits"] == []

def test_unknown_video_gives_404(client):
    assert client.get("/api/video/нет").status_code == 404

def test_timeline_returns_events(client):
    data = client.get("/api/timeline/abc123").json()

    assert data["duration"] == 60.0
    assert data["events"][0]["label"] == "sink"

def test_missing_index_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "INDEX", tmp_path / "нет.json")
    response = TestClient(api.app).get("/api/videos")

    assert response.status_code == 503
    assert "pipeline.py" in response.json()["detail"]

def test_timestamp_link_respects_existing_query():
    video = {"url": "https://www.youtube.com/watch?v=abc"}
    assert api.moment_url(video, 75) == "https://www.youtube.com/watch?v=abc&t=75s"

def test_local_video_has_no_timestamp_link():
    assert api.moment_url({"url": None}, 10) is None

# ---- CLI ----

def test_pipeline_defaults():
    args = pipeline.parse_args([])

    assert args.sources == "videos.txt"
    assert args.confidence == vision.CONFIDENCE
    assert args.whisper == speech.MODEL_SIZE

def test_sources_file_skips_comments(tmp_path):
    path = tmp_path / "videos.txt"
    path.write_text("# ссылки\nhttps://a\n\n  https://b  \n", encoding="utf-8")

    assert collect.read_sources(path) == ["https://a", "https://b"]
