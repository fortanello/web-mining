BOT_NAME = "webmining"

SPIDER_MODULES = ["webmining.spiders"]
NEWSPIDER_MODULE = "webmining.spiders"

# на стандартный User-Agent Scrapy многие сайты отвечают 403; только latin-1
USER_AGENT = "web-mining-lab4/1.0 (educational project; contact: student@example.com)"
ROBOTSTXT_OBEY = True

DOWNLOAD_DELAY = 2
DOWNLOAD_DELAY_JITTER = 0.5          # бывший RANDOMIZE_DOWNLOAD_DELAY
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 2
DOWNLOAD_TIMEOUT = 30

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [429, 500, 502, 503, 504]

ITEM_PIPELINES = {
    "webmining.pipelines.CleaningPipeline": 100,
    "webmining.pipelines.DuplicationPipeline": 200,
    "webmining.pipelines.ValidationPipeline": 300,
}

# Через браузер идут только запросы с meta={"playwright": True}, остальные — как обычно
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {"headless": True}
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 60_000

# без этого кириллица уедет в JSON как АБ
FEED_EXPORT_ENCODING = "utf-8"
FEED_EXPORT_INDENT = 2
LOG_LEVEL = "INFO"
