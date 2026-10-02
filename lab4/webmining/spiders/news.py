import scrapy
from itemloaders.processors import Join, MapCompose, TakeFirst
from scrapy.loader import ItemLoader

from webmining.collect import KeepCollecting
from webmining.items import NewsItem
from webmining.sites import NEWS_SITES

MAX_PAGES = 30

class NewsLoader(ItemLoader):
    default_item_class = NewsItem
    default_input_processor = MapCompose(str.strip)
    default_output_processor = TakeFirst()
    summary_out = Join(" ")

class NewsSpider(KeepCollecting, scrapy.Spider):
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
        self.page = 0
        self.exhausted = False

    # С версии 2.13 Scrapy вызывает async start(), а не start_requests()
    async def start(self):
        yield self.next_request()

    # KeepCollecting спрашивает следующую страницу, пока не наберётся limit записей
    def next_request(self):
        if self.exhausted or self.page >= MAX_PAGES:
            return None

        self.page += 1
        return scrapy.Request(self.profile["page_url"].format(page=self.page),
                              callback=self.parse, errback=self.on_error)

    def parse(self, response):
        cards = response.css(self.profile["card"])
        if not cards:
            # страницы кончились или сломался селектор — дальше листать бессмысленно
            self.logger.warning("Ни одной карточки на %s — проверьте селекторы в sites.py",
                                response.url)
            self.exhausted = True
            return

        for card in cards:
            item = self.build_item(card, response)
            if item is None:
                continue
            yield item

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

    def on_error(self, failure):
        self.logger.error("Страница %d не открылась: %s", self.page, failure.value)
        self.exhausted = True

    def closed(self, reason):
        if self.keywords:
            self.logger.info("Не подошло по ключевым словам — %d записей", self.skipped)
        super().closed(reason)
