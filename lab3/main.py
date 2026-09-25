import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

import graphviz

import wikimedia

ARTICLES_DEFAULT = 40
MIN_ARTICLES_REQUIRED = 30
OUTPUT_ROOT = Path(__file__).with_name("output")

SERVICE_PREFIXES = ("Категория:", "Шаблон:", "Файл:", "Служебная:", "Справка:",
                    "Портал:", "Проект:", "Википедия:", "Участник:", "Модуль:",
                    "Обсуждение", "Category:", "Template:", "File:", "Help:")
LIST_PREFIXES = ("Список ", "Перечень ", "List of ")

MONTHS = ("января|февраля|марта|апреля|мая|июня|июля|"
          "августа|сентября|октября|ноября|декабря")
DATE_TITLE = re.compile(rf"^\d{{3,4}}(\s+год|-е|$)|^\d{{1,2}}\s+({MONTHS})$|^[IVXLC]+\s+век")
FORBIDDEN_IN_FILENAME = re.compile(r'[<>:"/\\|?*]')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="wikimedia-collector",
        description="Сбор данных о предметной области через API проектов Wikimedia.",
        epilog='Пример: python main.py "Амурский тигр" --articles 40',
    )
    parser.add_argument("query", help="запрос: личность, объект, событие, животное")
    parser.add_argument("--lang", default="ru",
                        help="язык Википедии и Викисловаря (по умолчанию: ru)")
    parser.add_argument("--articles", type=int, default=ARTICLES_DEFAULT,
                        help=f"сколько статей собрать (по умолчанию: {ARTICLES_DEFAULT})")
    parser.add_argument("--images", type=int, default=10,
                        help="сколько изображений брать с Wikimedia Commons (по умолчанию: 10)")
    parser.add_argument("--skip-potd", action="store_true",
                        help="не скачивать картинку дня")
    return parser.parse_args(argv)

# Служебные страницы, даты и списки связывают статьи формально, а не по смыслу
def is_useful_article(title):
    if title.startswith(SERVICE_PREFIXES) or title.startswith(LIST_PREFIXES):
        return False
    return not DATE_TITLE.match(title)

def safe_filename(name):
    return FORBIDDEN_IN_FILENAME.sub("_", name).strip()

# Feed API дописывает к ссылке utm-метки, в имя файла они попасть не должны
def filename_from_url(url):
    return safe_filename(unquote(urlsplit(url).path.rsplit("/", 1)[-1]))

# Равномерная выборка по всему списку, иначе в граф попадут только статьи на первые буквы
def sample(items, size):
    step = max(1, len(items) // size)
    return items[::step][:size]

def collect_titles(host, root, limit):
    neighbours = [title for title in wikimedia.article_links(host, root)
                  if is_useful_article(title) and title != root]
    return [root] + sample(neighbours, limit - 1)

def collect_links(host, titles):
    inside = set(titles)
    links = {}

    for number, title in enumerate(titles, start=1):
        print(f"[{number:2}/{len(titles)}] {title}")
        targets = set(wikimedia.article_links(host, title)) & inside
        links[title] = sorted(targets - {title})

    return links

def isolated_nodes(links):
    connected = set()
    for source, targets in links.items():
        if targets:
            connected.add(source)
            connected.update(targets)
    return sorted(set(links) - connected)

def build_graph(root, links):
    graph = graphviz.Digraph("wikimedia", comment=f"Граф статей Википедии: {root}")
    graph.attr(rankdir="LR", overlap="false")
    graph.attr("node", shape="box", fontname="Arial", fontsize="10")

    for title in sorted(links):
        if title == root:
            graph.node(title, style="filled", fillcolor="#bfdcff", penwidth="2")
        else:
            graph.node(title)

    for source, targets in sorted(links.items()):
        for target in targets:
            mutual = source in links.get(target, ())
            if mutual and source > target:
                continue
            graph.edge(source, target, dir="both" if mutual else "forward")

    return graph

def save_graph(graph, folder):
    dot_path = folder / "graph.dot"
    graph.save(dot_path)
    print(f"Граф на языке DOT: {dot_path.name}")

    try:
        picture = graph.render(dot_path, format="svg", cleanup=False)
    except graphviz.ExecutableNotFound:
        print("Картинка не отрисована: нет программы dot, см. README.")
        return

    print(f"Картинка: {Path(picture).name}")

def collect_wiktionary(host, query):
    entries = []
    for word in query.split():
        definition = wikimedia.wiktionary_definition(host, word.lower())
        if definition:
            entries.append({"word": word.lower(), "definition": definition})
    return entries

def save_picture_of_the_day(folder, lang):
    picture = wikimedia.picture_of_the_day(date.today(), lang)
    if not picture.get("url"):
        print("Картинка дня недоступна.")
        return {}

    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / filename_from_url(picture["url"])

    saved = wikimedia.download(picture["url"], image_path, "potd-image")
    if not saved and picture.get("thumbnail"):
        # оригинал бывает на сотни мегабайт и отдаётся не всегда — берём превью
        saved = wikimedia.download(picture["thumbnail"], image_path, "potd-thumbnail")
    if not saved:
        print("Картинку дня скачать не удалось, подробности в errors.json")
        return {}

    # по заданию описание лежит рядом и называется так же, меняется только расширение
    text_path = image_path.with_suffix(".txt")
    text_path.write_text("\n".join([
        picture["title"],
        "",
        picture["description"],
        "",
        f"Автор: {picture['artist']}",
        f"Лицензия: {picture['license']}",
        f"Источник: {picture['file_page']}",
    ]), encoding="utf-8")

    print(f"Картинка дня: {image_path.name} (+ {text_path.name})")
    picture["saved_image"] = str(image_path)
    picture["saved_description"] = str(text_path)
    return picture

def build_result(query, root, content, links, langlinks, wikidata, wiktionary, images, picture):
    articles = []
    for title in sorted(content):
        article = dict(content[title])
        article["links"] = links.get(title, [])
        articles.append(article)

    return {
        "query": query,
        "collected_at": date.today().isoformat(),
        "root": {
            "title": root,
            "url": content.get(root, {}).get("url", ""),
            "extract": content.get(root, {}).get("extract", ""),
            "image": content.get(root, {}).get("image", ""),
            "language_versions": langlinks,
        },
        "wikidata": wikidata,
        "wiktionary": wiktionary,
        "commons_images": images,
        "picture_of_the_day": picture,
        "articles": articles,
        "graph": {
            "nodes": len(links),
            "edges": sum(len(targets) for targets in links.values()),
            "isolated": isolated_nodes(links),
        },
    }

def report(result):
    graph = result["graph"]
    print(f"\nСтатей в графе:  {graph['nodes']}")
    print(f"Связей:          {graph['edges']}")
    print(f"Изолированных:   {len(graph['isolated'])}")
    print(f"Изображений:     {len(result['commons_images'])}")
    print(f"Статей Викисловаря: {len(result['wiktionary'])}")
    print(f"Ошибок запросов: {len(wikimedia.errors)}")

    if graph["nodes"] < MIN_ARTICLES_REQUIRED:
        print(f"Внимание: по заданию нужно минимум {MIN_ARTICLES_REQUIRED} статей — "
              "увеличьте --articles.")
    if graph["isolated"]:
        print("Изолированные статьи (нет связей с остальными): "
              + ", ".join(graph["isolated"][:5]))

def main(argv=None):
    args = parse_args(argv)
    host = wikimedia.wikipedia_host(args.lang)

    root = wikimedia.search_article(host, args.query)
    if not root:
        sys.exit(f"По запросу «{args.query}» статья не найдена.")

    print(f"Корневая статья: {root}\n")
    titles = collect_titles(host, root, args.articles)
    links = collect_links(host, titles)
    content = wikimedia.articles_content(host, titles)

    langlinks = wikimedia.language_versions(host, root)
    wikidata = wikimedia.wikidata_entity(content.get(root, {}).get("wikidata_id", ""), args.lang)
    wiktionary = collect_wiktionary(wikimedia.wiktionary_host(args.lang), args.query)
    images = wikimedia.commons_images(args.query, args.images)

    folder = OUTPUT_ROOT / safe_filename(args.query.lower().replace(" ", "_"))
    folder.mkdir(parents=True, exist_ok=True)

    picture = {} if args.skip_potd else save_picture_of_the_day(folder / "potd", args.lang)

    result = build_result(args.query, root, content, links, langlinks,
                          wikidata, wiktionary, images, picture)
    (folder / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "errors.json").write_text(
        json.dumps(wikimedia.errors, ensure_ascii=False, indent=2), encoding="utf-8")

    save_graph(build_graph(root, links), folder)
    report(result)
    print(f"\nВсё сохранено в {folder}")
    return result

if __name__ == "__main__":
    main()
