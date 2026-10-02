import re

import scrapy
from itemloaders.processors import Identity, MapCompose, TakeFirst
from scrapy.loader import ItemLoader
from scrapy_playwright.page import PageMethod

from webmining.items import ProductItem
from webmining.sites import SHOP_SITES

SCROLL_STEPS = 12
SCROLL_PAUSE = 1000
PRICE = re.compile(r"\d[\d\s .,]*\d|\d")

class ProductLoader(ItemLoader):
    default_item_class = ProductItem
    default_input_processor = MapCompose(str.strip)
    default_output_processor = TakeFirst()
    # числа уже разобраны, strip к ним применять нельзя
    price_in = Identity()
    rating_in = Identity()
    reviews_in = Identity()

# «$1,139.54», «1 234,56 ₽», «от 89 990 ₽» — из всего этого нужна одна цифра
def parse_price(text):
    if not text:
        return None

    match = PRICE.search(text.replace(" ", " "))
    if not match:
        return None

    number = match.group().replace(" ", "")
    if "," in number and "." in number:
        number = number.replace(",", "")      # 1,139.54 — запятая разделяет тысячи
    else:
        number = number.replace(",", ".")     # 1234,56 — запятая отделяет копейки

    try:
        return float(number)
    except ValueError:
        return None

def parse_number(text):
    if not text:
        return None

    digits = re.sub(r"[^\d.]", "", text)
    return float(digits) if digits else None

# Товары подгружаются при прокрутке, поэтому сначала ждём карточки, потом листаем вниз
def page_methods(profile):
    methods = [PageMethod("wait_for_selector", profile["wait_for"])]
    for _ in range(SCROLL_STEPS):
        methods.append(PageMethod("evaluate", "window.scrollTo(0, document.body.scrollHeight)"))
        methods.append(PageMethod("wait_for_timeout", SCROLL_PAUSE))
    return methods

class EcommerceSpider(scrapy.Spider):
    name = "ecommerce"

    def __init__(self, site="testshop", category="", limit=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if site not in SHOP_SITES:
            raise ValueError(f"неизвестный магазин {site}, доступны: {', '.join(SHOP_SITES)}")

        self.site = site
        self.profile = SHOP_SITES[site]
        self.categories = [category] if category else self.profile["categories"]
        self.limit = int(limit)

    # С версии 2.13 Scrapy вызывает async start(), а не start_requests()
    async def start(self):
        for category in self.categories:
            yield scrapy.Request(
                self.profile["page_url"].format(category=category),
                meta={"playwright": True, "playwright_page_methods": page_methods(self.profile)},
                callback=self.parse,
                errback=self.on_error,
                dont_filter=True,
            )

    def parse(self, response):
        cards = response.css(self.profile["card"])
        if not cards:
            self.logger.warning("Ни одной карточки на %s — страница не отрендерилась "
                                "или изменились селекторы", response.url)
            return

        self.logger.info("Карточек на странице: %d (%s)", len(cards), response.url)
        for card in cards:
            yield self.build_item(card, response)

    def build_item(self, card, response):
        loader = ProductLoader(selector=card)
        loader.add_css("title", self.profile["title"])
        loader.add_value("source", self.site)
        loader.add_value("currency", self.profile["currency"])

        link = card.css(self.profile["url"]).get()
        loader.add_value("url", response.urljoin(link) if link else None)
        loader.add_value("price", parse_price(card.css(self.profile["price"]).get()))
        loader.add_value("rating", parse_number(card.css(self.profile["rating"]).get()))
        loader.add_value("reviews", parse_number(card.css(self.profile["reviews"]).get()))

        return loader.load_item()

    def on_error(self, failure):
        self.logger.error("Запрос не удался: %s", failure.value)
