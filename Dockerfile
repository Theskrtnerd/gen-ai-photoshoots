FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install poetry

RUN poetry install

EXPOSE 8501

LABEL maintainer="Theskrtnerd <tvbbd2@gmail.com>" \
      version="1.0"

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit", "run", "gen_ai_photoshoots/main.py", "--server.port=8501", "--server.address=0.0.0.0"]