from urllib.parse import urlsplit

import pytest
from scrapy.exceptions import DropItem
from scrapy import Request, signals
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DontCloseSpider
from scrapy.http import HtmlResponse

import run_spider
from webmining.items import NewsItem, ProductItem
from webmining.sites import NEWS_SITES, SHOP_SITES
from webmining.pipelines import CleaningPipeline, DuplicationPipeline, ValidationPipeline
from webmining.spiders.ecommerce import (EcommerceSpider, parse_number, parse_price,
                                         parse_rating)
from webmining.spiders import news as news_module
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

BOOKS_HTML = """
<html><body>
  <article class="product_pod">
    <div class="image_container">
      <a href="in-her-wake_980/index.html"><img src="thumb.jpg" class="thumbnail"></a>
    </div>
    <p class="star-rating One"><i class="icon-star"></i></p>
    <h3><a href="in-her-wake_980/index.html" title="In Her Wake">In Her Wake</a></h3>
    <div class="product_price">
      <p class="price_color">\u00a312.84</p>
      <p class="instock availability">In stock</p>
    </div>
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

# Scrapy отдаёт отклик, привязанный к запросу; магазин достаёт из meta свой профиль
def response_for(html, url, **meta):
    request = Request(url, meta=meta) if meta else None
    return HtmlResponse(url=url, body=html, encoding="utf-8", request=request)

def shop_response(html, url, site="testshop"):
    return response_for(html, url, site=site, profile=SHOP_SITES[site])

class FakeEngine:
    def __init__(self):
        self.requests = []

    def crawl(self, request):
        self.requests.append(request)

class FakeCrawler:
    def __init__(self):
        self.engine = FakeEngine()

# KeepCollecting подключается к сигналам, поэтому пауку нужен настоящий crawler
def spider_with_crawler(spider_class, **kwargs):
    crawler = CrawlerProcess(settings={}).create_crawler(spider_class)
    spider = spider_class.from_crawler(crawler, **kwargs)
    return spider, crawler

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
    items = list(spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/")))

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
    items = list(spider.parse(response_for(NEWS_HTML, "https://habr.com/ru/articles/page1/")))

    assert len(items) == 1
    assert "Нейросети" in items[0]["title"]
    assert spider.skipped == 1

def test_news_spider_survives_page_without_cards():
    spider = NewsSpider()
    items = list(spider.parse(response_for("<html><body></body></html>", "https://habr.com/")))

    assert items == []

def test_news_spider_walks_pages_one_by_one():
    spider, _ = spider_with_crawler(NewsSpider)

    assert "page1" in spider.next_request().url
    assert "page2" in spider.next_request().url

def test_news_spider_gives_up_after_max_pages():
    spider, _ = spider_with_crawler(NewsSpider)
    spider.page = news_module.MAX_PAGES

    assert spider.next_request() is None

def test_spider_asks_for_more_until_limit_reached():
    spider, _ = spider_with_crawler(NewsSpider, limit=100)
    spider.crawler = FakeCrawler()

    # DontCloseSpider — сигнал Scrapy «не закрывайся, есть ещё работа»
    with pytest.raises(DontCloseSpider):
        spider.ask_for_more()

    assert len(spider.crawler.engine.requests) == 1

def test_spider_keeps_going_past_the_minimum_to_leave_a_margin():
    spider, _ = spider_with_crawler(NewsSpider, limit=100)
    spider.crawler = FakeCrawler()
    spider.collected = 100

    # 100 — это минимум, а не цель: остановка ровно на границе оставляет ноль запаса
    with pytest.raises(DontCloseSpider):
        spider.ask_for_more()

    assert spider.target > spider.limit
    assert spider.enough

def test_spider_lets_itself_close_once_margin_collected():
    spider, _ = spider_with_crawler(NewsSpider, limit=100)
    spider.crawler = FakeCrawler()
    spider.collected = spider.target

    spider.ask_for_more()

    assert spider.crawler.engine.requests == []

def test_spider_lets_itself_close_when_sources_run_out():
    spider, _ = spider_with_crawler(EcommerceSpider, limit=100)
    spider.crawler = FakeCrawler()
    spider.queue = []

    spider.ask_for_more()

    assert spider.crawler.engine.requests == []

def test_running_out_of_sources_above_the_minimum_is_not_a_failure():
    spider, _ = spider_with_crawler(EcommerceSpider, limit=100)
    spider.crawler = FakeCrawler()
    spider.queue = []
    spider.collected = 105

    spider.ask_for_more()

    # запас набрать не вышло, но требование «не менее 100» выполнено
    assert spider.enough

def test_collected_counts_only_records_that_passed_pipelines():
    spider, crawler = spider_with_crawler(NewsSpider)

    crawler.signals.send_catch_log(signals.item_scraped, item={}, response=None, spider=spider)

    # send_catch_log глотает ошибки: ноль означал бы, что подпись обработчика не совпала
    assert spider.collected == 1

def test_ecommerce_spider_takes_asked_category_first():
    spider, _ = spider_with_crawler(EcommerceSpider, category="computers/tablets")

    assert spider.next_request().url.endswith("computers/tablets")

def test_ecommerce_spider_falls_back_to_other_shops():
    spider, _ = spider_with_crawler(EcommerceSpider)
    hosts = []
    while True:
        request = spider.next_request()
        if request is None:
            break
        hosts.append(urlsplit(request.url).netloc)

    assert len(hosts) > 10                       # одной категории на 100 товаров не хватит
    assert len(set(hosts)) > 1                   # запас лежит и на других сайтах
    assert "www.wildberries.ru" not in hosts     # антиботовый магазин в запас не берём

def test_ecommerce_spider_carries_profile_with_each_request():
    spider, _ = spider_with_crawler(EcommerceSpider)
    request = spider.next_request()

    # селекторы у магазинов разные, поэтому профиль едет вместе с запросом
    assert request.meta["site"] == "testshop"
    assert request.meta["profile"] is SHOP_SITES["testshop"]

def test_ecommerce_spider_scrolls_until_cards_stop_appearing():
    spider, _ = spider_with_crawler(EcommerceSpider)
    methods = spider.next_request().meta["playwright_page_methods"]

    assert [method.method for method in methods] == ["wait_for_selector", "evaluate"]
    assert "querySelectorAll" in methods[1].args[0]
    assert methods[1].args[1] == SHOP_SITES["testshop"]["card"]

@pytest.mark.parametrize("stars, expected", [
    ("star-rating Three", 3.0), ("star-rating One", 1.0), ("star-rating Five", 5.0),
])
def test_rating_written_as_a_word_becomes_a_number(stars, expected):
    assert parse_rating(stars, SHOP_SITES["books"]) == expected

def test_rating_written_as_a_number_stays_a_number():
    assert parse_rating("4", SHOP_SITES["testshop"]) == 4.0

def test_unknown_site_is_rejected():
    with pytest.raises(ValueError):
        NewsSpider(site="несуществующий")

def test_ecommerce_spider_parses_card():
    spider = EcommerceSpider()
    url = "https://webscraper.io/test-sites/e-commerce/scroll/computers/laptops"
    items = list(spider.parse(shop_response(SHOP_HTML, url)))

    assert len(items) == 1
    product = items[0]
    assert product["title"] == "Asus AsusPro"
    assert product["price"] == 1139.54
    assert product["rating"] == 3.0
    assert product["reviews"] == 14.0
    assert product["currency"] == "USD"
    assert product["url"] == "https://webscraper.io/test-sites/e-commerce/product/545"

def test_books_shop_card_is_parsed_with_its_own_selectors():
    spider = EcommerceSpider()
    url = "https://books.toscrape.com/catalogue/page-1.html"
    items = list(spider.parse(shop_response(BOOKS_HTML, url, site="books")))

    assert len(items) == 1
    book = items[0]
    assert book["title"] == "In Her Wake"
    assert book["price"] == 12.84              # цена с \u00a3, а не с $
    assert book["rating"] == 1.0               # рейтинг словом в классе
    assert book["currency"] == "GBP"
    assert book["source"] == "books"
    assert book["url"] == "https://books.toscrape.com/catalogue/in-her-wake_980/index.html"
    assert "reviews" not in book               # в этом магазине отзывов нет

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
