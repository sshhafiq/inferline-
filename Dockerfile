FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

COPY inferline/ ./inferline/
COPY config.example.yaml ./config.yaml

EXPOSE 8000

CMD ["python", "-m", "inferline.server"]
