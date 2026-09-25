import time

import requests

# Wikimedia требует User-Agent с названием инструмента и контактом, иначе отдаёт 403.
# Только latin-1: кириллица в заголовке роняет requests с UnicodeEncodeError
USER_AGENT = "web-mining-lab3/1.0 (educational project; contact: student@example.com)"

TIMEOUT = 20
PAUSE_BETWEEN_REQUESTS = 0.3
RETRY_ATTEMPTS = 3
RETRY_PAUSE = 5
TITLES_PER_REQUEST = 20

FEED_API = "https://api.wikimedia.org/feed/v1/wikipedia"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

errors = []

def wikipedia_host(lang):
    return f"{lang}.wikipedia.org"

def wiktionary_host(lang):
    return f"{lang}.wiktionary.org"

WIKIDATA_HOST = "www.wikidata.org"
COMMONS_HOST = "commons.wikimedia.org"

def fetch(url, params=None, stage=""):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            time.sleep(PAUSE_BETWEEN_REQUESTS)
            return response.json()
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", None)
            if status in (403, 429, 500, 502, 503) and attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_PAUSE * attempt)
                continue

            errors.append({"stage": stage, "url": url, "params": params,
                           "error": f"{type(error).__name__}: {status or error}"})
            return None

def action_api(host, stage, **params):
    params = {"action": "query", "format": "json", "formatversion": "2", **params}
    return fetch(f"https://{host}/w/api.php", params, stage or host)

def batched(items, size=TITLES_PER_REQUEST):
    for start in range(0, len(items), size):
        yield items[start:start + size]

def search_article(host, text):
    data = action_api(host, "search", list="search", srsearch=text, srlimit=1)
    results = (data or {}).get("query", {}).get("search", [])
    return results[0]["title"] if results else None

def article_links(host, title, limit=500):
    data = action_api(host, "links", titles=title, prop="links",
                      plnamespace=0, pllimit=limit)
    pages = (data or {}).get("query", {}).get("pages", [])
    if not pages or "links" not in pages[0]:
        return []
    return [link["title"] for link in pages[0]["links"]]

def articles_content(host, titles):
    content = {}
    for chunk in batched(titles):
        data = action_api(host, "content", titles="|".join(chunk),
                          prop="extracts|pageimages|info|pageprops",
                          exintro=1, explaintext=1, exlimit="max",
                          piprop="original", inprop="url", ppprop="wikibase_item")
        for page in (data or {}).get("query", {}).get("pages", []):
            if page.get("missing"):
                continue
            content[page["title"]] = {
                "title": page["title"],
                "url": page.get("fullurl", ""),
                "extract": page.get("extract", ""),
                "image": page.get("original", {}).get("source", ""),
                "wikidata_id": page.get("pageprops", {}).get("wikibase_item", ""),
            }
    return content

def language_versions(host, title, limit=20):
    data = action_api(host, "langlinks", titles=title, prop="langlinks",
                      lllimit=limit, llprop="url")
    pages = (data or {}).get("query", {}).get("pages", [])
    if not pages or "langlinks" not in pages[0]:
        return []
    return [{"lang": link["lang"], "title": link["title"], "url": link.get("url", "")}
            for link in pages[0]["langlinks"]]

def wikidata_entity(entity_id, lang="ru"):
    if not entity_id:
        return {}

    data = fetch(f"https://{WIKIDATA_HOST}/w/api.php", {
        "action": "wbgetentities", "format": "json", "formatversion": "2",
        "ids": entity_id, "languages": f"{lang}|en",
        "props": "labels|descriptions|aliases|claims|sitelinks/urls",
    }, "wikidata")

    entity = (data or {}).get("entities", {}).get(entity_id)
    if not entity:
        return {}

    labels = entity.get("labels", {})
    descriptions = entity.get("descriptions", {})
    return {
        "id": entity_id,
        "url": f"https://{WIKIDATA_HOST}/wiki/{entity_id}",
        "label": (labels.get(lang) or labels.get("en", {})).get("value", ""),
        "description": (descriptions.get(lang) or descriptions.get("en", {})).get("value", ""),
        "aliases": [alias["value"] for alias in entity.get("aliases", {}).get(lang, [])],
        "properties": wikidata_properties(entity.get("claims", {})),
        "sitelinks": len(entity.get("sitelinks", {})),
    }

def wikidata_properties(claims):
    properties = {}
    for code, statements in claims.items():
        values = []
        for statement in statements:
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(value, dict):
                value = value.get("id") or value.get("text") or value.get("time")
            if isinstance(value, str):
                values.append(value)
        if values:
            properties[code] = values[:5]
    return properties

def wiktionary_definition(host, word):
    data = action_api(host, "wiktionary", titles=word, prop="extracts",
                      explaintext=1, exsectionformat="plain")
    pages = (data or {}).get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return ""
    return pages[0].get("extract", "").strip()

def commons_images(text, limit=10):
    data = action_api(COMMONS_HOST, "commons", generator="search",
                      gsrsearch=text, gsrnamespace=6, gsrlimit=limit,
                      prop="imageinfo", iiprop="url|extmetadata")
    images = []
    for page in (data or {}).get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        images.append({
            "title": page["title"],
            "url": info.get("url", ""),
            "description_url": info.get("descriptionurl", ""),
            "description": strip_tags(meta.get("ImageDescription", {}).get("value", "")),
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "author": strip_tags(meta.get("Artist", {}).get("value", "")),
        })
    return images

def picture_of_the_day(day, lang="ru"):
    url = f"{FEED_API}/{lang}/featured/{day:%Y/%m/%d}"
    data = fetch(url, stage="potd") or {}
    image = data.get("image")

    if not image and lang != "en":
        data = fetch(f"{FEED_API}/en/featured/{day:%Y/%m/%d}", stage="potd-en") or {}
        image = data.get("image")

    if not image:
        return {}

    return {
        "title": image.get("title", ""),
        "url": image.get("image", {}).get("source", ""),
        "thumbnail": image.get("thumbnail", {}).get("source", ""),
        "description": strip_tags(image.get("description", {}).get("text", "")),
        "artist": strip_tags(image.get("artist", {}).get("text", "")),
        "license": image.get("license", {}).get("type", ""),
        "file_page": image.get("file_page", ""),
    }

def download(url, path, stage="download"):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            path.write_bytes(response.content)
            return True
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", None)
            if status in (403, 429, 500, 502, 503) and attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_PAUSE * attempt)
                continue

            errors.append({"stage": stage, "url": url,
                           "error": f"{type(error).__name__}: {status or error}"})
            return False

def strip_tags(html):
    if not html:
        return ""

    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)
