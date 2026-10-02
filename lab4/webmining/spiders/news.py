import scrapy
from itemloaders.processors import Join, MapCompose, TakeFirst
from scrapy import signals
from scrapy.loader import ItemLoader

from webmining.items import NewsItem
from webmining.sites import NEWS_SITES

MAX_PAGES = 30

class NewsLoader(ItemLoader):
    default_item_class = NewsItem
    default_input_processor = MapCompose(str.strip)
    default_output_processor = TakeFirst()
    summary_out = Join(" ")

class NewsSpider(scrapy.Spider):
    name = "news"

    def __init__(self, site="habr", keywords="", limit=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if site not in NEWS_SITES:
            raise ValueError(f"неизвестный сайт {site}, доступны: {', '.join(NEWS_SITES)}")

        self.site = site
        self.profile = NEWS_SITES[site]
        self.keywords = [word.strip().lower() for word in keywords.split(",") if word.strip()]
        self.limit = int(limit)
        self.skipped = 0
        self.collected = 0
        self.page = 1

    # item_scraped срабатывает только на записях, прошедших все pipelines,
    # поэтому отброшенные дубликаты не считаются собранными
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.count_item, signal=signals.item_scraped)
        return spider

    def count_item(self):
        self.collected += 1

    # С версии 2.13 Scrapy вызывает async start(), а не start_requests()
    async def start(self):
        yield self.page_request(self.page)

    def page_request(self, page):
        return scrapy.Request(self.profile["page_url"].format(page=page), callback=self.parse)

    def parse(self, response):
        cards = response.css(self.profile["card"])
        if not cards:
            self.logger.warning("Ни одной карточки на %s — проверьте селекторы в sites.py",
                                response.url)
            return

        for card in cards:
            item = self.build_item(card, response)
            if item is None:
                continue
            yield item

        # сколько страниц нужно, заранее неизвестно: зависит от того, сколько отсеет фильтр.
        # страницы идут строго по одной, поэтому номер хранится прямо в пауке
        if self.collected >= self.limit:
            return
        if self.page >= MAX_PAGES:
            self.logger.warning("Остановился на странице %d, собрано %d из %d — "
                                "ключевые слова слишком узкие",
                                self.page, self.collected, self.limit)
            return

        self.page += 1
        yield self.page_request(self.page)

    def build_item(self, card, response):
        loader = NewsLoader(selector=card)
        loader.add_css("title", self.profile["title"])
        loader.add_css("date", self.profile["date"])
        loader.add_css("summary", self.profile["summary"])
        loader.add_value("source", self.site)

        link = card.css(self.profile["url"]).get()
        loader.add_value("url", response.urljoin(link) if link else None)

        item = loader.load_item()
        if not self.matches_keywords(item):
            self.skipped += 1
            return None
        return item

    def matches_keywords(self, item):
        if not self.keywords:
            return True

        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        return any(word in text for word in self.keywords)

    def closed(self, reason):
        if self.keywords:
            self.logger.info("Не подошло по ключевым словам — %d записей", self.skipped)
        self.logger.info("Собрано записей: %d (нужно %d)", self.collected, self.limit)
