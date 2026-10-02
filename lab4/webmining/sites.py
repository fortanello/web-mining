NEWS_SITES = {
    "habr": {
        "page_url": "https://habr.com/ru/articles/page{page}/",
        "per_page": 20,
        "card": "article.tm-articles-list__item",
        "title": "h2 a span::text",
        "url": "h2 a::attr(href)",
        "date": "time::attr(datetime)",
        "summary": ".article-formatted-body ::text",
    },
    "lenta": {
        "page_url": "https://lenta.ru/parts/news/{page}/",
        "per_page": 20,
        "card": "a.card-full-news",
        "title": "h3::text",
        "url": "::attr(href)",
        "date": "time::attr(datetime)",
        "summary": ".card-full-news__info::text",
    },
}

SHOP_SITES = {
    "testshop": {
        "page_url": "https://webscraper.io/test-sites/e-commerce/scroll/computers/{category}",
        "categories": ["laptops", "tablets"],
        "wait_for": ".thumbnail",
        "card": ".thumbnail",
        "title": "a.title::attr(title)",
        "url": "a.title::attr(href)",
        "price": "h4.price::text",
        "rating": "p[data-rating]::attr(data-rating)",
        "reviews": "p.review-count::text",
        "currency": "USD",
    },
    # Заготовка для настоящего магазина. Wildberries закрыт антиботовой защитой:
    # паук, скорее всего, получит капчу и соберёт ноль записей — см. README
    "wildberries": {
        "page_url": "https://www.wildberries.ru/catalog/0/search.aspx?search={category}",
        "categories": ["ноутбуки"],
        "wait_for": ".product-card",
        "card": ".product-card",
        "title": ".product-card__name::text",
        "url": ".product-card__link::attr(href)",
        "price": ".price__lower-price::text",
        "rating": ".address-rate-mini::text",
        "reviews": ".product-card__count::text",
        "currency": "RUB",
    },
}
