import argparse
import sys
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from webmining.sites import NEWS_SITES, SHOP_SITES
from webmining.spiders.ecommerce import EcommerceSpider
from webmining.spiders.news import NewsSpider

OUTPUT_ROOT = Path(__file__).with_name("output")
LOG_ROOT = Path(__file__).with_name("logs")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_spider",
        description="Запуск пауков сбора данных с экспортом в JSON и CSV.",
        epilog='Пример: python run_spider.py --spider news --keywords "ИИ,нейросети" '
               '--output news.json',
    )
    parser.add_argument("--spider", required=True, choices=["news", "ecommerce"],
                        help="какого паука запускать")
    parser.add_argument("--output", required=True,
                        help="имя файла результата; рядом всегда пишутся и .json, и .csv")
    parser.add_argument("--keywords", default="",
                        help="слова через запятую: оставить только записи с ними (для news)")
    parser.add_argument("--category", default="",
                        help="категория товаров (для ecommerce)")
    parser.add_argument("--site", default="",
                        help=f"откуда собирать: {', '.join(NEWS_SITES)} для news, "
                             f"{', '.join(SHOP_SITES)} для ecommerce")
    parser.add_argument("--limit", type=int, default=100,
                        help="сколько записей собрать (по умолчанию 100)")
    return parser.parse_args(argv)

# Задание требует и JSON, и CSV, поэтому из имени берём только основу
def build_feeds(output):
    stem = OUTPUT_ROOT / Path(output).stem
    return {
        f"{stem}.json": {"format": "json", "overwrite": True},
        f"{stem}.csv": {"format": "csv", "overwrite": True},
    }

def main(argv=None):
    args = parse_args(argv)
    OUTPUT_ROOT.mkdir(exist_ok=True)
    LOG_ROOT.mkdir(exist_ok=True)

    settings = get_project_settings()
    settings.set("FEEDS", build_feeds(args.output))
    settings.set("LOG_FILE", str(LOG_ROOT / f"{args.spider}.log"))

    process = CrawlerProcess(settings)
    if args.spider == "news":
        process.crawl(NewsSpider, site=args.site or "habr",
                      keywords=args.keywords, limit=args.limit)
    else:
        process.crawl(EcommerceSpider, site=args.site or "testshop",
                      category=args.category, limit=args.limit)

    print(f"Паук {args.spider} запущен, лог: {LOG_ROOT / f'{args.spider}.log'}")
    process.start()

    for path in sorted(build_feeds(args.output)):
        size = Path(path).stat().st_size if Path(path).exists() else 0
        print(f"{Path(path).name}: {size} байт")

if __name__ == "__main__":
    sys.exit(main())
