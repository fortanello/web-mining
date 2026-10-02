import pytest
from scrapy.exceptions import DropItem
from scrapy import Request, signals
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse

import run_spider
from webmining.items import NewsItem, ProductItem
from webmining.sites import NEWS_SITES, SHOP_SITES
from webmining.pipelines import CleaningPipeline, DuplicationPipeline, ValidationPipeline
from webmining.spiders.ecommerce import EcommerceSpider, parse_number, parse_price
from webmining.spiders.news import NewsSpider

NEWS_HTML = """
<html><body>
  <article class="tm-articles-list__item">
    <h2><a href="/ru/articles/111/"><span>  Нейросети   в   металлургии </span></a></h2>
    <time datetime="2026-09-26T10:00:00.000Z">сегодня</time>
    <div class="article-formatted-body">Как ИИ помогает   доменным печам</div>
  </article>
  <article class="tm-articles-list__item">
    <h2><a href="/ru/articles/222/"><span>Разбор Kubernetes</span></a></h2>
    <time datetime="2026-09-25T10:00:00.000Z">вчера</time>
    <div class="article-formatted-body">Оркестрация контейнеров</div>
  </article>
</body></html>
"""

SHOP_HTML = """
<html><body>
  <div class="thumbnail">
    <h4 class="price">$1,139.54</h4>
    <h4><a href="/test-sites/e-commerce/product/545" class="title" title="Asus AsusPro">Asus</a></h4>
    <div class="ratings">
      <p class="review-count">14 reviews</p>
      <p data-rating="3"></p>
    </div>
  </div>
</body></html>
"""

def response_for(html, url):
    return HtmlResponse(url=url, body=html, encoding="utf-8")

# парсер новостей отдаёт и записи, и запрос следующей страницы — в тестах их разделяем
def items_only(results):
    return [result for result in results if not isinstance(result, Request)]

def requests_only(results):
    return [result for result in results if isinstance(result, Request)]

# ---- CleaningPipeline ----

def test_cleaning_fixes_broken_encoding():
    # классические кракозябры: utf-8 прочитали как latin-1
    broken = "металлургия".encode("utf-8").decode("latin-1")
    item = CleaningPipeline().process_item(NewsItem(title=broken))

    assert broken != "металлургия"
    assert item["title"] == "металлургия"

def test_cleaning_strips_and_collapses_spaces():
    pipeline = CleaningPipeline()
    item = pipeline.process_item(NewsItem(title="  Нейросети   в   металлургии "))

    assert item["title"] == "Нейросети в металлургии"

def test_cleaning_lowercases_only_configured_fields():
    pipeline = CleaningPipeline()
    item = pipeline.process_item(NewsItem(title="Большой Заголовок", summary="Текст ОПИСАНИЯ"))

    assert item["title"] == "Большой Заголовок"
    assert item["summary"] == "текст описания"

def test_cleaning_counts_changed_fields():
    pipeline = CleaningPipeline()
    pipeline.process_item(NewsItem(title=" пробелы "))
    pipeline.process_item(NewsItem(title="Чисто"))

    assert pipeline.changed == 1

# ---- DuplicationPipeline ----

def test_duplicate_url_is_dropped():
    pipeline = DuplicationPipeline()
    pipeline.process_item(NewsItem(title="а", url="https://site/1"))

    with pytest.raises(DropItem):
        pipeline.process_item(NewsItem(title="а", url="https://site/1"))

def test_same_product_with_tracking_parameters_is_one_record():
    pipeline = DuplicationPipeline()
    pipeline.process_item(ProductItem(title="Ноутбук", url="https://shop/p/1"))

    with pytest.raises(DropItem):
        pipeline.process_item(ProductItem(title="Ноутбук", url="https://shop/p/1?from=search"))

def test_without_url_duplicates_are_found_by_title():
    pipeline = DuplicationPipeline()
    pipeline.process_item(ProductItem(title="Ноутбук"))

    with pytest.raises(DropItem):
        pipeline.process_item(ProductItem(title="ноутбук"))

def test_different_records_pass():
    pipeline = DuplicationPipeline()
    pipeline.process_item(NewsItem(title="а", url="https://site/1"))
    pipeline.process_item(NewsItem(title="б", url="https://site/2"))

    assert pipeline.dropped == 0

# ---- ValidationPipeline ----

@pytest.mark.parametrize("item", [
    NewsItem(title="", url="https://site/1"),
    NewsItem(title="Заголовок"),
    ProductItem(title="Ноутбук", url="https://shop/1", price=0),
    ProductItem(title="Ноутбук", url="https://shop/1", price=99_000_000),
])
def test_invalid_records_are_dropped(item):
    with pytest.raises(DropItem):
        ValidationPipeline().process_item(item)

def test_valid_record_passes():
    item = ProductItem(title="Ноутбук", url="https://shop/1", price=1139.54)

    assert ValidationPipeline().process_item(item) is item

# ---- разбор цены и чисел ----

@pytest.mark.parametrize("text, expected", [
    ("$1,139.54", 1139.54),
    ("1 234,56 ₽", 1234.56),
    ("от 89 990 ₽", 89990.0),
    ("1139.54", 1139.54),
    ("нет в наличии", None),
    ("", None),
    (None, None),
])
def test_parse_price(text, expected):
    assert parse_price(text) == expected

@pytest.mark.parametrize("text, expected", [("14 reviews", 14.0), ("3", 3.0), ("", None)])
def test_parse_number(text, expected):
    assert parse_number(text) == expected

# ---- пауки ----

def test_news_spider_extracts_fields_and_absolute_url():
    spider = NewsSpider()
    items = items_only(spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/")))

    assert len(items) == 2
    assert "Нейросети" in items[0]["title"]
    assert items[0]["url"] == "https://habr.com/ru/articles/111/"
    assert items[0]["date"].startswith("2026-09-26")
    assert items[0]["source"] == "habr"

def test_spider_and_pipelines_together_give_clean_record():
    spider = NewsSpider()
    raw = next(iter(spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/"))))

    item = CleaningPipeline().process_item(raw)
    item = DuplicationPipeline().process_item(item)
    item = ValidationPipeline().process_item(item)

    assert item["title"] == "Нейросети в металлургии"
    assert item["summary"] == "как ии помогает доменным печам"

def test_news_spider_keeps_only_matching_keywords():
    spider = NewsSpider(keywords="нейросети,ии")
    items = items_only(spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/")))

    assert len(items) == 1
    assert "Нейросети" in items[0]["title"]
    assert spider.skipped == 1

def test_news_spider_survives_page_without_cards():
    spider = NewsSpider()
    items = list(spider.parse(response_for("<html><body></body></html>", "https://habr.com/")))

    assert items == []

def test_news_spider_asks_for_next_page_while_below_limit():
    spider = NewsSpider(limit=100)
    requests = requests_only(
        spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/")))

    assert len(requests) == 1
    assert "page2" in requests[0].url
    assert spider.page == 2

def test_news_spider_stops_paging_when_limit_reached():
    spider = NewsSpider(limit=2)
    spider.collected = 2
    requests = requests_only(
        spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/")))

    assert requests == []
    assert spider.page == 1

def test_news_spider_counts_only_records_that_passed_pipelines():
    crawler = CrawlerProcess(settings={}).create_crawler(NewsSpider)
    spider = NewsSpider.from_crawler(crawler)

    crawler.signals.send_catch_log(signals.item_scraped, item={}, response=None, spider=spider)

    # send_catch_log глотает ошибки: ноль означал бы, что подпись обработчика не совпала
    assert spider.collected == 1

def test_unknown_site_is_rejected():
    with pytest.raises(ValueError):
        NewsSpider(site="несуществующий")

def test_ecommerce_spider_parses_card():
    spider = EcommerceSpider()
    url = "https://webscraper.io/test-sites/e-commerce/scroll/computers/laptops"
    items = list(spider.parse(response_for(SHOP_HTML, url)))

    assert len(items) == 1
    product = items[0]
    assert product["title"] == "Asus AsusPro"
    assert product["price"] == 1139.54
    assert product["rating"] == 3.0
    assert product["reviews"] == 14.0
    assert product["currency"] == "USD"
    assert product["url"] == "https://webscraper.io/test-sites/e-commerce/product/545"

# ---- CLI ----

def test_both_formats_are_exported():
    feeds = run_spider.build_feeds("news.json")
    formats = {settings["format"] for settings in feeds.values()}

    assert formats == {"json", "csv"}
    assert all(path.endswith((".json", ".csv")) for path in feeds)

def test_output_extension_does_not_matter():
    assert set(run_spider.build_feeds("prices.csv")) == set(run_spider.build_feeds("prices.json"))

def test_cli_requires_spider_and_output():
    with pytest.raises(SystemExit):
        run_spider.parse_args(["--spider", "news"])

@pytest.mark.parametrize("profile", list(NEWS_SITES.values()) + list(SHOP_SITES.values()))
def test_profile_separates_page_address_from_link_selector(profile):
    assert profile["page_url"].startswith("https://")
    assert "::" in profile["url"]
