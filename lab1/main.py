import argparse
import json
import shlex
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

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

cdx_toolkit.myrequests.MAX_ERRORS = 3  # по умолчанию 100 повторов по минуте

cache = {}  # всё, что уже скачано из индекса и WARC: повторный запуск в сеть не ходит

# Аргументы командной строки (argv — для запуска из тетрадки)
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="cc-search",
        description="Поиск страниц в архиве Common Crawl по CDX-индексу.",
        epilog="Пример: python main.py Пастернак --domain ru.wikipedia.org --text Перм",
    )
    parser.add_argument(
        "keywords",
        nargs="*",
        help="слова, которые должны быть в URL страницы, с учётом регистра (без слов — все страницы)",
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        default=["pstu.ru"],
        help="один или несколько сайтов (по умолчанию: pstu.ru)",
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
        help="оставить страницы, в тексте которых есть все эти слова",
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
def search(domain, keywords, limit):
    key = f"search {domain} {' '.join(keywords)} {limit}"
    if key in cache:
        return cache[key]

    found = []
    seen = set()  # один и тот же URL лежит сразу в нескольких обходах
    # статус, тип и слова в адресе проверяет сам сервер индекса; ~ — «содержит»
    filters = ["=status:200", "=mime:text/html"] + ["~url:" + quote(word) for word in keywords]
    try:
        # в сеть лезут оба вызова: за списком обходов и за страницами индекса
        fetcher = cdx_toolkit.CDXFetcher(source="cc", crawl=CRAWL)
        for scanned, capture in enumerate(fetcher.iter(f"{domain}/*", filter=filters), start=1):
            if capture["url"] not in seen:
                seen.add(capture["url"])
                found.append(dict(capture))
            if len(found) >= limit or scanned >= SCAN_LIMIT:
                break
    except Exception as error:
        print(f"Индекс ответил ошибкой ({type(error).__name__}), показываю что успел найти.")
        return found  # неполный ответ в кэш не кладём

    cache[key] = found
    return found

# Загрузка страницы из WARC: Range-запрос по offset из индекса
def load_page(record):
    key = f"page {record['filename']} {record['offset']}"
    if key in cache:
        return cache[key]

    capture = cdx_toolkit.CaptureObject(record, warc_download_prefix="https://data.commoncrawl.org")
    try:
        html = capture.content.decode(record.get("charset") or "utf-8", errors="replace")
    except Exception:
        return ["—", "(не удалось загрузить)"]

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()  # иначе в текст попадёт код
    title = soup.title.get_text(strip=True) if soup.title else "—"
    content = soup.find("main") or soup.body or soup  # у Википедии статья лежит в <main>, без меню
    cache[key] = [title, content.get_text(" ", strip=True)]
    return cache[key]

# Есть ли в тексте все слова (без учёта регистра)
def has_words(text, words):
    text = text.lower()
    return all(word.lower() in text for word in words)

# Кусок текста вокруг первого найденного слова — в нём виден контекст упоминания
def fragment(text, words):
    lowered = text.lower()
    positions = [lowered.find(word.lower()) for word in words if word.lower() in lowered]
    start = max(0, min(positions) - 50) if positions else 0
    return text[start:start + FRAGMENT_LENGTH]

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
    texts = [load_page(record)[1].lower() for record in records]
    rows = []
    for word in words:
        word = word.lower()
        rows.append([word, sum(word in text for text in texts), sum(text.count(word) for text in texts)])
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
        print(f"Поиск в {domain}, слова в адресе: {args.keywords or 'не заданы'}...")
        # для проверки текста страниц берём с запасом: часть отсеется
        found = search(domain, args.keywords, TEXT_LIMIT if words else args.limit)
        if args.text:
            found = [record for record in found if has_words(load_page(record)[1], args.text)]
        records += found
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    rows = [build_row(record, words, show_text) for record in records[:args.limit]]
    print_table(rows, show_text, len(records))
    if args.count:
        print_counts(records, args.count)
    if args.stats and records:
        print_stats(records)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

# Запуск из тетрадки: run("Пастернак --domain ru.wikipedia.org") — то же, что python main.py ...
def run(command):
    main(shlex.split(command))

if __name__ == "__main__":
    main()
