NEWS_SITES = {
    "habr": {
        "page_url": "https://habr.com/ru/articles/page{page}/",
        "card": "article.tm-articles-list__item",
        "title": "h2 a span::text",
        "url": "h2 a::attr(href)",
        "date": "time::attr(datetime)",
        "summary": ".article-formatted-body ::text",
    },
    "lenta": {
        "page_url": "https://lenta.ru/parts/news/{page}/",
        "card": "a.card-full-news",
        "title": "h3::text",
        "url": "::attr(href)",
        "date": "time::attr(datetime)",
        "summary": ".card-full-news__info::text",
    },
}

SHOP_SITES = {
    # Каталог webscraper.io сделан для тренировки скрапинга: настоящий SPA, без капчи
    # и без запрета в правилах. Разметка во всех его разделах одинаковая.
    "testshop": {
        "page_url": "https://webscraper.io/test-sites/e-commerce/scroll/{category}",
        "categories": ["computers/laptops", "computers/tablets", "phones/touch"],
        "wait_for": ".thumbnail",
        "card": ".thumbnail",
        "title": "a.title::attr(title)",
        "url": "a.title::attr(href)",
        "price": "h4.price::text",
        "rating": "p[data-rating]::attr(data-rating)",
        "reviews": "p.review-count::text",
        "currency": "USD",
    },
    # Тот же каталог, но товары догружаются запросами, а не прокруткой
    "testshop-ajax": {
        "page_url": "https://webscraper.io/test-sites/e-commerce/ajax/{category}",
        "categories": ["computers/laptops", "computers/tablets", "phones/touch"],
        "wait_for": ".thumbnail",
        "card": ".thumbnail",
        "title": "a.title::attr(title)",
        "url": "a.title::attr(href)",
        "price": "h4.price::text",
        "rating": "p[data-rating]::attr(data-rating)",
        "reviews": "p.review-count::text",
        "currency": "USD",
    },
    # Книжный магазин-песочница: 1000 товаров по 20 на страницу, рейтинг словом в классе
    "books": {
        "page_url": "https://books.toscrape.com/catalogue/{category}.html",
        "categories": [f"page-{number}" for number in range(1, 11)],
        "wait_for": "article.product_pod",
        "card": "article.product_pod",
        "title": "h3 a::attr(title)",
        "url": "h3 a::attr(href)",
        "price": "p.price_color::text",
        "rating": "p.star-rating::attr(class)",
        "rating_words": {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5},
        "currency": "GBP",
    },
    # Заготовка для настоящего магазина. Wildberries закрыт антиботовой защитой: паук,
    # скорее всего, получит капчу и соберёт ноль записей, поэтому в запас его не берём
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
        "antibot": True,
    },
}
