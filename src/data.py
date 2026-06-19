"""Загрузка сплитов и подготовка пар (текст, группа)."""
import json
import re
from pathlib import Path

from .mapping import group_of, is_dropped

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_WS = re.compile(r"\s+")


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def load_xy(
    name: str, *, drop_excluded: bool = True, normalize_text: bool = False
) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for row in load_split(name):
        code = row["code"]
        if drop_excluded and is_dropped(code):
            continue
        text = row["symptoms"]
        texts.append(normalize(text) if normalize_text else text)
        labels.append(group_of(code))
    return texts, labels
