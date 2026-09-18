BOT_NAME = "quotes"

SPIDER_MODULES = ["quotes.spiders"]
NEWSPIDER_MODULE = "quotes.spiders"

# на стандартный User-Agent Scrapy многие сайты отвечают 403
USER_AGENT = "lab2-quotes-spider (educational project)"

ROBOTSTXT_OBEY = True

DOWNLOAD_DELAY = 0.5
AUTOTHROTTLE_ENABLED = True
CONCURRENT_REQUESTS_PER_DOMAIN = 4
DOWNLOAD_TIMEOUT = 20
RETRY_TIMES = 3

# без этого кириллица уедет в JSON как АБ
FEED_EXPORT_ENCODING = "utf-8"
FEED_EXPORT_INDENT = 2
