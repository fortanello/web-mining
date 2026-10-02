import logging
from urllib.parse import urlsplit

import ftfy
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

TEXT_FIELDS = ("title", "summary")
LOWERCASE_FIELDS = ("summary", "source")
REQUIRED_FIELDS = ("title", "url")
MAX_PRICE = 10_000_000

# Начиная со Scrapy 2.13 в process_item и close_spider не передаётся spider,
# поэтому логируем через обычный логгер модуля
logger = logging.getLogger(__name__)

class CleaningPipeline:
    def __init__(self):
        self.changed = 0

    def process_item(self, item):
        adapter = ItemAdapter(item)

        for field in TEXT_FIELDS:
            value = adapter.get(field)
            if not isinstance(value, str):
                continue

            clean = " ".join(ftfy.fix_text(value).split())
            if field in LOWERCASE_FIELDS:
                clean = clean.lower()
            if clean != value:
                self.changed += 1
            adapter[field] = clean

        return item

    def close_spider(self):
        logger.info("CleaningPipeline: поправлено полей — %d", self.changed)

class DuplicationPipeline:
    def __init__(self):
        self.seen = set()
        self.dropped = 0

    def process_item(self, item):
        key = duplicate_key(ItemAdapter(item))
        if key in self.seen:
            self.dropped += 1
            raise DropItem(f"дубликат: {key}")

        self.seen.add(key)
        return item

    def close_spider(self):
        logger.info("DuplicationPipeline: отброшено дубликатов — %d", self.dropped)

# Один товар встречается на нескольких страницах с разными метками в адресе,
# поэтому ключ — адрес без параметров запроса, а если адреса нет, то заголовок
def duplicate_key(adapter):
    url = (adapter.get("url") or "").strip().lower()
    if url:
        return urlsplit(url)._replace(query="", fragment="").geturl()
    return (adapter.get("title") or "").strip().lower()

class ValidationPipeline:
    def __init__(self):
        self.dropped = 0

    def process_item(self, item):
        adapter = ItemAdapter(item)

        for field in REQUIRED_FIELDS:
            if not adapter.get(field):
                self.dropped += 1
                raise DropItem(f"нет поля {field}: {adapter.get('url') or adapter.get('title')}")

        price = adapter.get("price")
        if price is not None and not 0 < price < MAX_PRICE:
            self.dropped += 1
            raise DropItem(f"цена вне разумных границ ({price}): {adapter.get('url')}")

        return item

    def close_spider(self):
        logger.info("ValidationPipeline: отброшено записей — %d", self.dropped)
