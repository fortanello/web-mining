import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import graphviz
import requests
from bs4 import BeautifulSoup

# rutermextract тянет pymorphy2, а тот падает на Python 3.11+ (зовёт удалённый
# inspect.getargspec). pymorphy3 — живой форк с тем же API, подставляем его
# под именем pymorphy2 до импорта rutermextract
import pymorphy3

sys.modules["pymorphy2"] = pymorphy3

from rutermextract import TermExtractor  # noqa: E402

TIMEOUT = 15
PAUSE_BETWEEN_REQUESTS = 1.0
RETRY_ATTEMPTS = 3
RETRY_PAUSE = 5
MIN_TEXT_LENGTH = 500
MIN_SOURCES_REQUIRED = 25
TEXT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
JUNK_TAGS = ["script", "style", "nav", "header", "footer", "aside"]

# Wikimedia требует User-Agent с названием инструмента и контактом, иначе отвечает 403.
# Только latin-1: кириллица в заголовке роняет requests с UnicodeEncodeError
HEADERS = {"User-Agent": "lab2-metallurgy-graph/1.0 (educational project; contact: student@example.com)"}

SOURCES_FILE = Path(__file__).with_name("sources.txt")
TERMS_FILE = Path(__file__).with_name("terms.json")
DOT_FILE = Path(__file__).with_name("metallurgy_graph.dot")

extractor = TermExtractor()
session = requests.Session()
session.headers.update(HEADERS)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="metallurgy-graph",
        description="Граф веб-ресурсов по металлургии: связи через общие ключевые термины.",
        epilog="Пример: python main.py --min-common 4 --top-terms 30",
    )
    parser.add_argument("--sources", default=SOURCES_FILE,
                        help="файл со списком URL (по умолчанию: sources.txt рядом со скриптом)")
    parser.add_argument("--top-terms", type=int, default=20,
                        help="сколько ключевых терминов брать с ресурса (по умолчанию: 20)")
    parser.add_argument("--min-common", type=int, default=3,
                        help="сколько общих терминов даёт ребро между ресурсами (по умолчанию: 3)")
    parser.add_argument("--offline", action="store_true",
                        help="не скачивать заново, взять термины из terms.json")
    return parser.parse_args(argv)

def read_sources(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        sys.exit(f"Не найден файл со списком ресурсов: {path}")

    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

def site_name(url):
    parsed = urlparse(url)
    name = parsed.netloc.replace("www.", "")
    path_parts = [part for part in parsed.path.split("/") if part]
    return f"{name}/{unquote(path_parts[-1])}" if path_parts else name

def download(url):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            # не text: кодировку русских страниц надёжнее определит BeautifulSoup
            return response.content
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", None)
            if status in (403, 429) and attempt < RETRY_ATTEMPTS:
                wait = RETRY_PAUSE * attempt
                print(f"    {status}, ждём {wait} с и пробуем снова")
                time.sleep(wait)
                continue

            print(f"    пропускаю — {status or type(error).__name__}")
            return None

def extract_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(JUNK_TAGS):
        tag.decompose()

    parts = [tag.get_text(" ", strip=True) for tag in soup.find_all(TEXT_TAGS)]
    return " ".join(part for part in parts if part)

def extract_terms(text, top_terms):
    return {term.normalized for term in extractor(text, limit=top_terms)}

def collect_terms(urls, top_terms):
    terms_by_site = load_terms()

    for number, url in enumerate(urls, start=1):
        print(f"[{number}/{len(urls)}] {url}")

        if site_name(url) in terms_by_site:
            print("    уже собран, пропускаю")
            continue

        html = download(url)
        if html is None:
            continue

        text = extract_text(html)
        if len(text) < MIN_TEXT_LENGTH:
            print(f"    пропускаю — мало текста ({len(text)} символов)")
            continue

        terms = extract_terms(text, top_terms)
        terms_by_site[site_name(url)] = terms
        print(f"    терминов: {len(terms)}")
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    return terms_by_site

def save_terms(terms_by_site):
    data = {site: sorted(terms) for site, terms in terms_by_site.items()}
    TERMS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nТермины сохранены: {TERMS_FILE.name}")

def load_terms():
    if not TERMS_FILE.exists():
        return {}

    data = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    return {site: set(terms) for site, terms in data.items()}

def find_links(terms_by_site, min_common):
    sites = sorted(terms_by_site)
    links = []

    for position, first in enumerate(sites):
        for second in sites[position + 1:]:
            common = terms_by_site[first] & terms_by_site[second]
            if len(common) >= min_common:
                links.append((first, second, sorted(common)))

    return links

def build_graph(terms_by_site, links):
    graph = graphviz.Digraph("metallurgy", comment="Граф веб-ресурсов по металлургии")
    graph.attr(rankdir="LR")
    graph.attr("node", shape="box", fontname="Arial")

    for site, terms in sorted(terms_by_site.items()):
        graph.node(site)
        graph.body.append(f"\t// {site}: {', '.join(sorted(terms)[:8])}\n")

    for first, second, common in links:
        graph.edge(first, second, label=str(len(common)))

    return graph

def save_graph(graph):
    graph.save(DOT_FILE)
    print(f"Граф на языке DOT: {DOT_FILE.name}")

    try:
        picture = graph.render(DOT_FILE, format="svg", cleanup=False)
    except graphviz.ExecutableNotFound:
        print("Картинка не отрисована: нет программы dot, см. README.")
        return

    print(f"Картинка: {Path(picture).name}")

def main(argv=None):
    args = parse_args(argv)

    if args.offline:
        terms_by_site = load_terms()
        if not terms_by_site:
            sys.exit(f"Нет файла {TERMS_FILE.name} — сначала запустите без --offline.")
        print(f"Термины взяты из {TERMS_FILE.name}: ресурсов {len(terms_by_site)}")
    else:
        terms_by_site = collect_terms(read_sources(args.sources), args.top_terms)
        save_terms(terms_by_site)

    if not terms_by_site:
        sys.exit("Не удалось собрать ни одного ресурса — проверьте список и интернет.")

    links = find_links(terms_by_site, args.min_common)
    print(f"\nРесурсов в графе: {len(terms_by_site)}")
    print(f"Связей между ними: {len(links)}")

    if len(terms_by_site) < MIN_SOURCES_REQUIRED:
        print(f"Внимание: по заданию нужно минимум {MIN_SOURCES_REQUIRED} ресурсов — "
              "добавьте ссылок в sources.txt.")
    if not links:
        print("Связей нет — уменьшите --min-common или увеличьте --top-terms.")

    save_graph(build_graph(terms_by_site, links))

if __name__ == "__main__":
    main()
