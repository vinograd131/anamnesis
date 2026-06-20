"""LLM-генерация НОВЫХ жалоб через Mistral REST API (синтетические примеры для слабых классов)."""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests

from .data import load_split
from .mapping import description_of, group_of, is_dropped

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-small-latest"

SYSTEM = (
    "Ты генерируешь правдоподобные жалобы пациентов на русском языке для аугментации "
    "медицинского датасета. Пиши РАЗНЫЕ, новые жалобы: разные пациенты, разные наборы "
    "симптомов в рамках диагноза, разный стиль (разговорный и клинический). Жалобы должны "
    "быть реалистичными, как в настоящих анамнезах. НЕ копируй примеры дословно."
)


def generate(api_key: str, code: str, desc: str, examples: list[str], n: int, retries: int = 4) -> list[str]:
    style = "\n".join(f"- {e}" for e in examples[:3])
    prompt = (
        f"Диагноз: {desc} (код {code}).\n"
        f"Примеры реальных жалоб (только для стиля):\n{style}\n\n"
        f"Сгенерируй {n} НОВЫХ, непохожих друг на друга и на примеры жалоб пациентов с этим "
        f'состоянием. Верни JSON-объект вида {{"complaints": ["...", "..."]}}.'
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
            return json.loads(content).get("complaints", [])[:n]
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} через {wait}s ({exc})", flush=True)
            time.sleep(wait)
    return []


def main(
    target_per_class: int = 500,
    per_call: int = 5,
    delay: float = 1.0,
    groups: list[str] | None = None,
) -> None:
    api_key = os.environ["MISTRAL_API_KEY"]
    rows = [r for r in load_split("train") if not is_dropped(r["code"])]

    by_group: dict[str, list[dict]] = defaultdict(list)
    by_code: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_group[group_of(r["code"])].append(r)
        by_code[r["code"]].append(r["symptoms"])

    targets = set(groups) if groups else None
    out_rows = list(rows)
    gi = 0
    for group, items in sorted(by_group.items()):
        if targets is not None and group not in targets:
            continue
        need = target_per_class - len(items)
        if need <= 0:
            print(f"{group:18s} {len(items):4d} -> без генерации", flush=True)
            continue
        codes = sorted({r["code"] for r in items})
        print(f"{group:18s} {len(items):4d} -> генерирую +{need} по {len(codes)} диагнозам", flush=True)
        made, ci = 0, 0
        while made < need:
            code = codes[ci % len(codes)]
            batch = generate(
                api_key, code, description_of(code), by_code[code], min(per_call, need - made)
            )
            for text in batch:
                out_rows.append({"idx": f"gen{gi}", "symptoms": text, "code": code})
                gi += 1
                made += 1
            ci += 1
            print(f"  {group}: {made}/{need}", flush=True)
            time.sleep(delay)

    path = DATA_DIR / "train_gen_v1.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nсохранено: {path}  (всего {len(out_rows)} строк)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--per-call", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--groups", default=None, help="список групп через запятую")
    args = parser.parse_args()
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else None
    main(args.target, args.per_call, args.delay, groups)
