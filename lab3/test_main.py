import pytest

import main
import wikimedia

@pytest.mark.parametrize("title", [
    "Амурский тигр", "Красная книга России", "Уссурийская тайга", "Panthera tigris",
])
def test_useful_articles_pass(title):
    assert main.is_useful_article(title)

@pytest.mark.parametrize("title", [
    "Категория:Кошачьи", "Шаблон:Таксон", "Файл:Tiger.jpg", "Служебная:Поиск",
    "Обсуждение:Тигр", "Портал:Биология", "Template:Taxonomy",
    "1947 год", "2022", "1990-е", "15 сентября", "XIX век",
    "Список млекопитающих России", "List of mammals",
])
def test_junk_titles_filtered(title):
    assert not main.is_useful_article(title)

def test_safe_filename_removes_windows_forbidden_characters():
    assert main.safe_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"

def test_filename_from_url_drops_query_and_decodes():
    url = ("https://upload.wikimedia.org/wikipedia/commons/1/1a/Lac%20Bab.jpg"
           "?utm_source=commons.wikimedia.org&utm_campaign=imageinfo")

    assert main.filename_from_url(url) == "Lac Bab.jpg"

def test_filename_from_url_stays_windows_safe():
    url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Panorama%3A%20Volga.jpg"

    name = main.filename_from_url(url)
    assert name == "Panorama_ Volga.jpg"
    assert (main.Path("/tmp") / name).with_suffix(".txt").name == "Panorama_ Volga.txt"

def test_sample_spreads_over_whole_list():
    items = list(range(100))
    chosen = main.sample(items, 10)
    assert len(chosen) == 10
    assert chosen[0] == 0
    assert chosen[-1] >= 90

def test_sample_returns_everything_when_list_is_short():
    assert main.sample([1, 2, 3], 10) == [1, 2, 3]

def test_collect_titles_puts_root_first_and_drops_junk(monkeypatch):
    monkeypatch.setattr(wikimedia, "article_links",
                        lambda host, title, limit=500: ["Категория:Кошачьи", "Тигр", "2022", "Тайга"])

    titles = main.collect_titles("ru.wikipedia.org", "Амурский тигр", 10)

    assert titles[0] == "Амурский тигр"
    assert "Тигр" in titles and "Тайга" in titles
    assert "2022" not in titles and "Категория:Кошачьи" not in titles

def test_collect_links_keeps_only_links_inside_the_set(monkeypatch):
    pages = {"А": ["Б", "Снаружи"], "Б": ["А"], "В": []}
    monkeypatch.setattr(wikimedia, "article_links", lambda host, title, limit=500: pages[title])

    links = main.collect_links("ru.wikipedia.org", ["А", "Б", "В"])

    assert links == {"А": ["Б"], "Б": ["А"], "В": []}

def test_isolated_nodes_found():
    links = {"А": ["Б"], "Б": [], "В": []}
    assert main.isolated_nodes(links) == ["В"]

def test_mutual_link_becomes_one_double_headed_edge():
    source = main.build_graph("А", {"А": ["Б"], "Б": ["А"]}).source

    assert source.count("->") == 1
    assert "dir=both" in source

def test_one_way_link_stays_directed():
    source = main.build_graph("А", {"А": ["Б"], "Б": []}).source

    assert source.count("->") == 1
    assert "dir=both" not in source

def test_root_is_highlighted():
    assert "fillcolor" in main.build_graph("А", {"А": [], "Б": []}).source

def test_result_has_every_section_the_task_asks_for():
    content = {"А": {"title": "А", "url": "u", "extract": "текст", "image": "i", "wikidata_id": "Q1"}}
    result = main.build_result("запрос", "А", content, {"А": []}, [], {}, [], [], {})

    assert set(result) >= {"query", "root", "wikidata", "wiktionary", "commons_images",
                           "picture_of_the_day", "articles", "graph"}
    assert result["articles"][0]["links"] == []
    assert result["graph"]["nodes"] == 1

def test_wikidata_properties_keeps_only_readable_values():
    claims = {
        "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q16521"}}}}],
        "P18": [{"mainsnak": {"datavalue": {"value": "Tiger.jpg"}}}],
        "P625": [{"mainsnak": {"datavalue": {"value": {"latitude": 1}}}}],
    }
    properties = wikimedia.wikidata_properties(claims)

    assert properties["P31"] == ["Q16521"]
    assert properties["P18"] == ["Tiger.jpg"]
    assert "P625" not in properties

def test_strip_tags_removes_markup():
    assert wikimedia.strip_tags("<div>Тигр <i>амурский</i></div>") == "Тигр амурский"

def test_batched_splits_titles():
    assert list(wikimedia.batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
