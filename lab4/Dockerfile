FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright тянет браузер и его системные библиотеки одной командой
RUN playwright install --with-deps chromium

COPY . .

CMD ["python", "run_spider.py", "--spider", "ecommerce", "--output", "products.json"]
