# CPU-инференс сервиса классификации фитнес-ограничений.
FROM python:3.13-slim

WORKDIR /app

# CPU-сборка torch (без CUDA) — лёгкий образ, запускается без GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Код, LoRA-адаптер и данные (нужны для baseline-fallback).
COPY src/ ./src/
COPY models/rubioroberta_ft/ ./models/rubioroberta_ft/
COPY data/ ./data/

# Вшиваем базовую модель в образ, чтобы контейнер работал офлайн (вариант A).
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    AutoTokenizer.from_pretrained('alexyalunin/RuBioRoBERTa'); \
    AutoModelForSequenceClassification.from_pretrained('alexyalunin/RuBioRoBERTa', num_labels=8)"

ENV MODEL=transformer
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
