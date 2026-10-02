import math

from scrapy import signals
from scrapy.exceptions import DontCloseSpider

# Задание требует НЕ МЕНЕЕ limit записей. Останавливаться ровно на границе нельзя: один
# новый дубликат или поправленная вёрстка — и записей окажется меньше требуемого. Поэтому
# паук добирает с запасом, а нехваткой считается только результат ниже limit.
MARGIN = 1.2

# Сколько нужно страниц или категорий, заранее неизвестно: фильтр по словам, дедупликация
# и валидация отбрасывают часть. Решение о следующем запросе принимается в spider_idle:
# к этому моменту очередь пуста, все записи уже прошли pipelines и посчитаны.
class KeepCollecting:
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.count_item, signal=signals.item_scraped)
        crawler.signals.connect(spider.ask_for_more, signal=signals.spider_idle)
        return spider

    @property
    def target(self):
        return math.ceil(self.limit * MARGIN)

    @property
    def enough(self):
        return self.collected >= self.limit

    def count_item(self):
        self.collected += 1

    def ask_for_more(self):
        if self.collected >= self.target:
            return

        request = self.next_request()
        if request is None:
            self.report_shortage()
            return

        self.crawler.engine.crawl(request)
        raise DontCloseSpider

    def report_shortage(self):
        if self.enough:
            self.logger.info("Источники закончились: запас набрать не удалось, но требование "
                             "выполнено — %d при минимуме %d", self.collected, self.limit)
        else:
            self.logger.warning("Источники закончились: собрано %d, а нужно не менее %d",
                                self.collected, self.limit)

    def closed(self, reason):
        self.logger.info("Собрано записей: %d (нужно не менее %d, требование выполнено: %s)",
                         self.collected, self.limit, "да" if self.enough else "НЕТ")
