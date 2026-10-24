"""Статическая временная шкала для отчёта. Интерактивная — на странице интерфейса."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"object": "#3fb950", "speech": "#58a6ff", "loud": "#d29922", "scene": "#bc8cff"}
ROWS = {"object": 3, "speech": 2, "loud": 1, "scene": 0}
TITLES = {"object": "объекты", "speech": "речь", "loud": "громко", "scene": "сцены"}


def plot(video, path, max_label=28):
    events = video["events"]
    duration = video.get("duration") or max((item["end"] for item in events), default=1)

    figure, axis = plt.subplots(figsize=(13, 4.2))
    for event in events:
        row = ROWS.get(event["type"], 0)
        width = max(event["end"] - event["start"], duration * 0.004)
        # сцены идут вплотную, поэтому им нужна светлая граница, иначе сольются
        edge = "white" if event["type"] == "scene" else "none"
        axis.barh(row, width, left=event["start"], height=0.65,
                  color=COLORS.get(event["type"], "#8b949e"),
                  edgecolor=edge, linewidth=1.2)

    # подписываем только события по объектам: речь и сцены сольются в кашу.
    # Объекты, начинающиеся в одно время, разводим по высоте
    taken = {}
    for event in sorted(events, key=lambda item: item["start"]):
        if event["type"] != "object":
            continue
        spot = round(event["start"] / max(duration, 1) * 40)
        shift = taken.get(spot, 0)
        taken[spot] = shift + 1
        axis.text(event["start"], ROWS["object"] + 0.42 + shift * 0.26,
                  event["label"][:max_label], fontsize=8, color="#30363d")

    axis.set_yticks(list(ROWS.values()))
    axis.set_yticklabels([TITLES[name] for name in ROWS])
    axis.set_xlim(0, duration)
    axis.set_ylim(-0.6, ROWS["object"] + 1.2)
    axis.set_xlabel("Время, с")
    axis.set_title(f"События: {video['title'][:70]}")
    axis.grid(axis="x", alpha=0.3)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
