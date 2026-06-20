"""Стресс-тест на робастность: перефразируем test через Mistral для проверки устойчивости моделей.

Перефразированный test (data/test_para_v1.jsonl) — ОЦЕНОЧНЫЙ набор, на нём не обучаемся.
Идея: tf-idf чувствителен к точным словам и проседает на перефразировках, семантический
трансформер — почти нет. Метки (диагнозы) сохраняются.
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

from .data import load_split
from .mapping import is_dropped

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-small-latest"

SYSTEM = (
    "Перефразируй жалобу пациента на русском другими словами, СОХРАНИВ медицинский смысл "
    "и подразумеваемый диагноз. Используй разговорные формы, синонимы, возможные опечатки. "
    "НЕ добавляй новых симптомов и не меняй суть."
)


def paraphrase_one(api_key: str, text: str, retries: int = 4) -> str:
    prompt = (
        f"Жалоба: {text}\n\n"
        f'Перефразируй её. Верни JSON-объект вида {{"paraphrase": "..."}}.'
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                raise RuntimeError("rate limit (429)")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content).get("paraphrase", "").strip()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} через {wait}s ({exc})", flush=True)
            time.sleep(wait)
    return ""


def main(delay: float = 0.5, limit: int | None = None) -> None:
    api_key = os.environ["MISTRAL_API_KEY"]
    rows = [r for r in load_split("test") if not is_dropped(r["code"])]
    if limit:
        rows = rows[:limit]

    out_rows = []
    for i, r in enumerate(rows):
        para = paraphrase_one(api_key, r["symptoms"])
        out_rows.append({"idx": r["idx"], "symptoms": para or r["symptoms"], "code": r["code"]})
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        time.sleep(delay)

    path = DATA_DIR / "test_para_v1.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nсохранено: {path}  ({len(out_rows)} перефразированных жалоб)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.delay, args.limit)
