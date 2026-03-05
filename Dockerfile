FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir -e .

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pyutagent", "--help"]
