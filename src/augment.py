"""LLM-аугментация train: перефразирование жалоб через Mistral REST API (только train, слабые классы)."""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests

from .data import load_split
from .mapping import group_of, is_dropped

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-small-latest"

SYSTEM = (
    "Ты помогаешь аугментировать датасет медицинских жалоб на русском языке. "
    "Перефразируй жалобу пациента разными способами, сохраняя медицинский смысл "
    "и подразумеваемое состояние. Меняй формулировки, порядок слов, разговорные "
    "обороты и возможные опечатки, но НЕ добавляй новых симптомов и не меняй суть."
)


def paraphrase(api_key: str, text: str, group: str, n: int, retries: int = 4) -> list[str]:
    prompt = (
        f"Группа ограничений: {group}.\n"
        f"Жалоба: {text}\n\n"
        f"Сгенерируй {n} разных перефразировок этой жалобы. "
        f'Верни JSON-объект вида {{"paraphrases": ["...", "..."]}}.'
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
            return json.loads(content).get("paraphrases", [])[:n]
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} через {wait}s ({exc})", flush=True)
            time.sleep(wait)
    return []


def main(
    target_per_class: int = 400,
    per_call: int = 5,
    delay: float = 1.0,
    groups: list[str] | None = None,
) -> None:
    api_key = os.environ["MISTRAL_API_KEY"]
    rows = [r for r in load_split("train") if not is_dropped(r["code"])]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[group_of(r["code"])].append(r)

    targets = set(groups) if groups else None
    out_rows = list(rows)
    for group, items in sorted(by_class.items()):
        if targets is not None and group not in targets:
            continue
        need = target_per_class - len(items)
        if need <= 0:
            print(f"{group:18s} {len(items):4d} -> без аугментации", flush=True)
            continue
        print(f"{group:18s} {len(items):4d} -> генерирую +{need}...", flush=True)
        made, i = 0, 0
        while made < need:
            src = items[i % len(items)]
            batch = paraphrase(api_key, src["symptoms"], group, min(per_call, need - made))
            for para in batch:
                out_rows.append(
                    {"idx": f"{src['idx']}_aug{made}", "symptoms": para, "code": src["code"]}
                )
                made += 1
            i += 1
            print(f"  {group}: {made}/{need}", flush=True)
            time.sleep(delay)

    path = DATA_DIR / "train_aug_v1.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nсохранено: {path}  (всего {len(out_rows)} строк)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--per-call", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--groups", default=None, help="список групп через запятую")
    args = parser.parse_args()
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else None
    main(args.target, args.per_call, args.delay, groups)
