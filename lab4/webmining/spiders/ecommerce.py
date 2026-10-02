import re

import scrapy
from itemloaders.processors import Identity, MapCompose, TakeFirst
from scrapy.loader import ItemLoader
from scrapy_playwright.page import PageMethod

from webmining.collect import KeepCollecting
from webmining.items import ProductItem
from webmining.sites import SHOP_SITES

SCROLL_STEPS = 40
SCROLL_PAUSE = 700
PRICE = re.compile(r"\d[\d\s .,]*\d|\d")

# Сколько прокруток нужно, зависит от каталога, поэтому листаем, пока число карточек растёт
SCROLL = """
async (selector) => {
  let previous = 0;
  for (let step = 0; step < STEPS; step++) {
    window.scrollTo(0, document.body.scrollHeight);
    await new Promise(resolve => setTimeout(resolve, PAUSE));
    const count = document.querySelectorAll(selector).length;
    if (count === previous) break;
    previous = count;
  }
  return previous;
}
""".replace("STEPS", str(SCROLL_STEPS)).replace("PAUSE", str(SCROLL_PAUSE))

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

# В одних магазинах рейтинг — число в атрибуте, в других слово в классе: «star-rating Three»
def parse_rating(text, profile):
    words = profile.get("rating_words")
    if not words or not text:
        return parse_number(text)

    for word, value in words.items():
        if word in text.split():
            return float(value)
    return None

class EcommerceSpider(KeepCollecting, scrapy.Spider):
    name = "ecommerce"

    def __init__(self, site="testshop", category="", limit=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if site not in SHOP_SITES:
            raise ValueError(f"неизвестный магазин {site}, доступны: {', '.join(SHOP_SITES)}")

        self.limit = int(limit)
        self.collected = 0
        self.queue = self.build_queue(site, category)

    # Одного магазина на limit может не хватить, поэтому за запрошенным идут остальные:
    # каждый источник — это магазин, его профиль и категория
    def build_queue(self, site, category):
        profile = SHOP_SITES[site]
        names = [category] if category else profile["categories"]
        queue = [(site, profile, name) for name in names]

        for other, spare in SHOP_SITES.items():
            if other == site or spare.get("antibot"):
                continue
            queue += [(other, spare, name) for name in spare["categories"]]
        return queue

    # С версии 2.13 Scrapy вызывает async start(), а не start_requests()
    async def start(self):
        yield self.next_request()

    # KeepCollecting спрашивает следующую категорию, пока не наберётся limit записей
    def next_request(self):
        if not self.queue:
            return None

        site, profile, category = self.queue.pop(0)
        return scrapy.Request(
            profile["page_url"].format(category=category),
            meta={
                "playwright": True,
                "playwright_page_methods": [
                    PageMethod("wait_for_selector", profile["wait_for"]),
                    PageMethod("evaluate", SCROLL, profile["card"]),
                ],
                "site": site,
                "profile": profile,
            },
            callback=self.parse,
            errback=self.on_error,
            dont_filter=True,
        )

    def parse(self, response):
        profile = response.meta["profile"]
        cards = response.css(profile["card"])
        if not cards:
            # источник не отдал товаров — паук возьмёт следующий из запаса
            self.logger.warning("Ни одной карточки на %s — страница не отрендерилась "
                                "или изменились селекторы", response.url)
            return

        self.logger.info("Карточек на странице: %d (%s)", len(cards), response.url)
        for card in cards:
            yield self.build_item(card, response)

    def build_item(self, card, response):
        profile, site = response.meta["profile"], response.meta["site"]

        loader = ProductLoader(selector=card)
        loader.add_css("title", profile["title"])
        loader.add_value("source", site)
        loader.add_value("currency", profile["currency"])

        link = card.css(profile["url"]).get()
        loader.add_value("url", response.urljoin(link) if link else None)
        loader.add_value("price", parse_price(card.css(profile["price"]).get()))
        loader.add_value("rating", parse_rating(card.css(profile["rating"]).get(), profile))
        if profile.get("reviews"):
            loader.add_value("reviews", parse_number(card.css(profile["reviews"]).get()))

        return loader.load_item()

    def on_error(self, failure):
        self.logger.error("Запрос не удался: %s", failure.value)
