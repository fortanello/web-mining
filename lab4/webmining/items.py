import scrapy

class NewsItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    date = scrapy.Field()
    summary = scrapy.Field()
    source = scrapy.Field()

class ProductItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    description = scrapy.Field()
    price = scrapy.Field()
    currency = scrapy.Field()
    rating = scrapy.Field()
    reviews = scrapy.Field()
    source = scrapy.Field()
