"""Загрузка датасетов из практики 2 и подготовка текстов к векторизации."""
import json
import re
from pathlib import Path

import pymorphy3

# Поля, из которых собирается документ для каждого датасета
DATASETS = {
    "news": ("news.json", ("title", "summary")),
    "products": ("prices.json", ("title", "description")),
}

WORD = re.compile(r"[а-яёa-z]+")
MIN_WORD_LENGTH = 3

# Список собран из NLTK и дополнен тем, что лезет в топики на веб-данных:
# слова статейных шаблонов, единицы измерения, обрывки разметки
STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "всё",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только",
    "ее", "её", "мне", "было", "вот", "от", "меня", "еще", "ещё", "нет", "о", "из", "ему",
    "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя",
    "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под",
    "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "неё", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед",
    "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно",
    "всю", "между", "который", "которая", "которые", "которых", "свой", "весь", "такие",
    # английские служебные слова: в корпусе товаров есть англоязычные описания книг
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "as",
    "by", "at", "from", "is", "are", "was", "were", "be", "been", "being", "am", "it", "its",
    "this", "that", "these", "those", "you", "your", "yours", "we", "our", "ours", "they",
    "their", "theirs", "them", "he", "she", "his", "her", "hers", "him", "i", "me", "my",
    "not", "no", "nor", "can", "will", "would", "should", "could", "shall", "may", "might",
    "must", "has", "have", "had", "having", "do", "does", "did", "doing", "there", "here",
    "when", "where", "why", "how", "what", "which", "who", "whom", "whose", "all", "any",
    "both", "each", "few", "more", "most", "other", "others", "some", "such", "only", "own",
    "same", "than", "then", "too", "very", "just", "also", "about", "above", "after", "again",
    "against", "before", "below", "between", "during", "into", "through", "under", "until",
    "up", "down", "out", "off", "over", "once", "one", "two", "three", "now", "ever", "never",
    "because", "while", "so", "s", "t", "don", "didn", "doesn", "isn", "wasn", "aren",
    # шаблонные обороты веб-страниц и единицы
    "дата", "обращения", "статья", "читать", "далее", "подробнее", "комментарий", "источник",
    "который", "также", "однако", "свои", "очень", "нужно", "делать", "сделать", "просто",
    "см", "мм", "кг", "шт", "руб", "год", "года", "лет", "день", "время", "раза",
    # общеупотребительные глаголы и местоимения: после лемматизации лезут в топ тем
    "это", "мочь", "стать", "хотеть", "знать", "говорить", "самый", "каждый", "любой",
    "наш", "ваш", "весь", "вообще", "вместе", "часто", "обычно", "например",
}

morph = pymorphy3.MorphAnalyzer()
_lemma_cache = {}


def lemma(word):
    if word not in _lemma_cache:
        _lemma_cache[word] = morph.parse(word)[0].normal_form
    return _lemma_cache[word]


# Текст -> список лемм без стоп-слов и коротких слов
def tokenize(text):
    tokens = []
    for word in WORD.findall(text.lower()):
        if len(word) < MIN_WORD_LENGTH or word in STOPWORDS:
            continue
        normal = lemma(word)
        if len(normal) >= MIN_WORD_LENGTH and normal not in STOPWORDS:
            tokens.append(normal)
    return tokens


def load_records(dataset, data_dir):
    if dataset == "mixed":
        records = []
        for name in ("news", "products"):
            records.extend(load_records(name, data_dir))
        return records

    if dataset not in DATASETS:
        raise ValueError(f"неизвестный датасет {dataset}, доступны: "
                         f"{', '.join(DATASETS)}, mixed")

    filename, fields = DATASETS[dataset]
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"нет файла {path}: сначала соберите данные практикой 2")

    rows = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in rows:
        text = " ".join(str(row.get(field, "")) for field in fields).strip()
        if not text:
            continue
        records.append({
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "source": row.get("source", dataset),
            "text": text,
        })
    return records


# Документы, у которых после очистки осталось слишком мало слов, только мешают моделям
def prepare(dataset, data_dir, min_tokens=5):
    records = load_records(dataset, data_dir)

    documents = []
    for record in records:
        tokens = tokenize(record["text"])
        if len(tokens) < min_tokens:
            continue
        record["tokens"] = tokens
        documents.append(record)

    dropped = len(records) - len(documents)
    return documents, {"loaded": len(records), "dropped": dropped, "kept": len(documents)}
