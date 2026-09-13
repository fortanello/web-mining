import argparse
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cdx_toolkit
from bs4 import BeautifulSoup
from tabulate import tabulate

# 16 последних обходов. Только списком: строку "16" cdx-toolkit разберёт посимвольно
CRAWL = ["16"]
SCAN_LIMIT = 5000      # предел записей индекса на домен, иначе на большом домене поиск не кончится
TEXT_LIMIT = 40        # сколько страниц с домена скачивать, когда нужен текст (--text, --count)
FRAGMENT_LENGTH = 200  # сколько символов текста страницы показываем
CACHE_FILE = Path(__file__).with_name("cache.json")
HEADERS = ["URL", "Дата архивации", "Заголовок", "Фрагмент текста"]
COLUMN_WIDTHS = [50, 14, 30, 40]
HIGHLIGHT, RESET = "\033[1;31m", "\033[0m"  # жирный красный для найденных слов

cdx_toolkit.myrequests.MAX_ERRORS = 3  # по умолчанию 100 повторов по минуте

cache = {}  # всё, что уже скачано из индекса и WARC: повторный запуск в сеть не ходит

# Аргументы командной строки (argv — для запуска из тетрадки)
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="cc-search",
        description="Поиск страниц в архиве Common Crawl по CDX-индексу.",
        epilog="Пример: python main.py --domain ru.wikipedia.org --prefix wiki/Пастернак --text Перм*",
    )
    parser.add_argument(
        "keywords",
        nargs="*",
        help="слова, которые должны быть в URL страницы, без учёта регистра (без слов — все страницы)",
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        default=["pstu.ru"],
        help="один или несколько сайтов (по умолчанию: pstu.ru)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="начало пути на сайте, с учётом регистра (например: wiki/Пастернак)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="сколько результатов показать (по умолчанию: 10)",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        default=[],
        help="оставить страницы, в тексте которых есть все эти слова (Перм* — слова на «Перм»)",
    )
    parser.add_argument(
        "--count",
        nargs="+",
        default=[],
        help="посчитать, на скольких страницах и сколько раз встречается каждое слово",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="загрузить страницу из WARC и показать заголовок и фрагмент текста",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="распределение найденных страниц по доменам и датам",
    )
    return parser.parse_args(argv)

# Поиск по CDX-индексу: URL, дата, offset. WARC не трогаем
def search(domain, prefix, keywords, limit):
    query = f"{domain}/{prefix.strip('/')}*"  # начало адреса сервер отбирает сам
    key = f"search {query} {' '.join(keywords)} {limit}"
    if key in cache:
        return cache[key]

    found = []
    seen = set()  # один и тот же URL лежит сразу в нескольких обходах
    filters = ["=status:200", "=mime:text/html"]
    try:
        # в сеть лезут оба вызова: за списком обходов и за страницами индекса
        fetcher = cdx_toolkit.CDXFetcher(source="cc", crawl=CRAWL)
        for scanned, capture in enumerate(fetcher.iter(query, filter=filters), start=1):
            if url_matches(capture["url"], keywords) and capture["url"] not in seen:
                seen.add(capture["url"])
                found.append(dict(capture))
            if len(found) >= limit or scanned >= SCAN_LIMIT:
                break
    except Exception as error:
        print(f"Индекс ответил ошибкой ({type(error).__name__}), показываю что успел найти.")
        return found  # неполный ответ в кэш не кладём

    cache[key] = found
    return found

def url_matches(url, keywords):
    url = unquote(url).lower()
    return all(word.lower().rstrip("*") in url for word in keywords)

# Равномерная выборка: не первые страницы по алфавиту, а через одинаковый шаг по всему списку
def sample(records, size):
    step = max(1, len(records) // size)
    return records[::step][:size]

# Загрузка страницы из WARC: Range-запрос по offset из индекса
def load_page(record):
    key = f"page {record['filename']} {record['offset']}"
    if key in cache:
        return cache[key]

    capture = cdx_toolkit.CaptureObject(record, warc_download_prefix="https://data.commoncrawl.org")
    try:
        html = capture.content
    except Exception:
        return ["—", "(не удалось загрузить)"]

    # байты, а не строка: не все сайты в UTF-8, кодировку BeautifulSoup найдёт сам
    soup = BeautifulSoup(html, "html.parser", from_encoding=record.get("charset"))
    title = soup.title.get_text(strip=True) if soup.title else "—"
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()  # код и меню в текст не нужны
    # у Википедии текст статьи лежит в #mw-content-text, на других сайтах часто в <main>
    content = soup.find(id="mw-content-text") or soup.find("main") or soup.body or soup
    cache[key] = [title, content.get_text(" ", strip=True)]
    return cache[key]

# Слово ищется целиком без учёта регистра, а «Перм*» — любое слово, начинающееся на «Перм».
# Поэтому «МГУ» не найдётся внутри «СамГУПС»
def word_pattern(word):
    if word.endswith("*"):
        return re.compile(r"\b" + re.escape(word[:-1]) + r"\w*", re.IGNORECASE)
    return re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)

# Есть ли в тексте все слова
def has_words(text, words):
    return all(word_pattern(word).search(text) for word in words)

# Кусок текста вокруг первого найденного слова, найденные слова выделены цветом
def fragment(text, words):
    matches = [word_pattern(word).search(text) for word in words]
    positions = [match.start() for match in matches if match]
    start = max(0, min(positions) - 50) if positions else 0
    piece = text[start:start + FRAGMENT_LENGTH]
    for word in words:
        piece = word_pattern(word).sub(lambda match: HIGHLIGHT + match.group(0) + RESET, piece)
    return piece

# Дата из CDX в читаемый вид: 20250131120000 -> 2025-01-31
def format_date(timestamp):
    return f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"

# Склейка строки таблицы
def build_row(record, words, show_text):
    row = [unquote(record["url"]), format_date(record["timestamp"])]  # иначе одни %-коды

    if show_text:
        title, text = load_page(record)
        row += [title, fragment(text, words)]
    else:
        row.append("— (нужен --show-text)")

    return row

# Вывод таблицы
def print_table(rows, show_text, total):
    if not rows:
        print("Ничего не найдено. Попробуйте другой домен или другие ключевые слова.")
        return

    columns = 4 if show_text else 3  # без --show-text заголовка и текста просто нет
    print(
        tabulate(
            rows,
            headers=HEADERS[:columns],
            tablefmt="grid",
            maxcolwidths=COLUMN_WIDTHS[:columns],
        )
    )
    print(f"\nНайдено результатов: {total}, показано: {len(rows)}")

# Сколько страниц упоминают каждое слово и сколько всего упоминаний
def print_counts(records, words):
    texts = [load_page(record)[1] for record in records]
    rows = []
    for word in words:
        pattern = word_pattern(word)
        found = [len(pattern.findall(text)) for text in texts]
        rows.append([word, sum(1 for count in found if count), sum(found)])
    print(f"\nСтраниц проверено: {len(texts)}")
    print(tabulate(rows, headers=["Слово", "Страниц", "Упоминаний"], tablefmt="grid"))

# Распределение по доменам и по месяцам архивации
def print_stats(records):
    domains = Counter(urlsplit(record["url"]).hostname.removeprefix("www.") for record in records)
    months = Counter(format_date(record["timestamp"])[:7] for record in records)
    print()
    print(tabulate(domains.most_common(), headers=["Домен", "Страниц"], tablefmt="grid"))
    print()
    print(tabulate(sorted(months.items()), headers=["Месяц архивации", "Страниц"], tablefmt="grid"))

def main(argv=None):
    args = parse_args(argv)
    words = args.text + args.count
    show_text = args.show_text or bool(words)
    if CACHE_FILE.exists():
        cache.update(json.loads(CACHE_FILE.read_text(encoding="utf-8")))

    records = []
    for domain in args.domain:
        print(f"Поиск в {domain}/{args.prefix.strip('/')}*, слова в адресе: {args.keywords or 'не заданы'}...")
        if words:
            # текст качать долго: из всего найденного берём равномерную выборку
            found = sample(search(domain, args.prefix, args.keywords, SCAN_LIMIT), TEXT_LIMIT)
        else:
            found = search(domain, args.prefix, args.keywords, args.limit)
        if args.text:
            found = [record for record in found if has_words(load_page(record)[1], args.text)]
        records += found
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    rows = [build_row(record, words + args.keywords, show_text) for record in records[:args.limit]]
    print_table(rows, show_text, len(records))
    if args.count:
        print_counts(records, args.count)
    if args.stats and records:
        print_stats(records)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

def run(command):
    main(shlex.split(command))

if __name__ == "__main__":
    main()
