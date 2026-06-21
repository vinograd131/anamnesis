"""REST-сервис: жалоба пациента -> группа ограничений + рекомендация.

Прод-модель — RuBioRoBERTa файнтюн (MODEL=transformer, по умолчанию). Для лёгкого
локального запуска без torch есть fallback на baseline (MODEL=baseline).

Эндпоинты:
  GET  /ping     — проверка живости
  GET  /health   — статус + какая модель загружена
  GET  /groups   — все группы и их ограничения
  POST /predict  — {"text": "..."} -> {group, restriction, confidence}
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .mapping import DISCLAIMER, GROUPS, RESTRICTIONS

MODEL_KIND = os.environ.get("MODEL", "transformer")
MAX_LENGTH = 256

_state: dict = {}


def _load_predictor():
    """Возвращает (predict_proba(texts)->matrix, classes) для выбранной модели."""
    if MODEL_KIND == "baseline":
        from .behavioral import baseline_predictor

        return baseline_predictor()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .transformer_ft import (
        ID2LABEL,
        LABEL2ID,
        MODEL_ID,
        MODELS,
        NAME,
        _predict_logits,
        _softmax,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(GROUPS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    model = PeftModel.from_pretrained(base, str(MODELS / NAME))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict_proba(texts):
        return _softmax(_predict_logits(model, tokenizer, texts, device, max_length=MAX_LENGTH))

    return predict_proba, list(GROUPS)


def _get_predictor():
    if "fn" not in _state:
        _state["fn"], _state["classes"] = _load_predictor()
    return _state["fn"], _state["classes"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_predictor()  # прогреваем модель при старте, чтобы первый запрос не ждал
    yield


app = FastAPI(title="anamnes — классификатор фитнес-ограничений", lifespan=lifespan)


class Complaint(BaseModel):
    text: str = Field(..., min_length=1, examples=["болит поясница, отдаёт в ногу"])


class Prediction(BaseModel):
    group: str
    restriction: str
    confidence: float
    disclaimer: str = DISCLAIMER


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KIND, "loaded": "fn" in _state}


@app.get("/groups")
def groups():
    return [{"group": g, "restriction": RESTRICTIONS[g]} for g in GROUPS]


@app.post("/predict", response_model=Prediction)
def predict(item: Complaint):
    predict_proba, classes = _get_predictor()
    proba = predict_proba([item.text])[0]
    idx = int(proba.argmax())
    group = classes[idx]
    return Prediction(
        group=group,
        restriction=RESTRICTIONS[group],
        confidence=round(float(proba[idx]), 4),
    )
