"""LLM-аугментация train: перефразирование жалоб через Mistral API (только train, слабые классы)."""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from mistralai import Mistral

from .data import load_split
from .mapping import group_of, is_dropped

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL = "mistral-small-latest"

SYSTEM = (
    "Ты помогаешь аугментировать датасет медицинских жалоб на русском языке. "
    "Перефразируй жалобу пациента разными способами, сохраняя медицинский смысл "
    "и подразумеваемое состояние. Меняй формулировки, порядок слов, разговорные "
    "обороты и возможные опечатки, но НЕ добавляй новых симптомов и не меняй суть."
)


def paraphrase(client: Mistral, text: str, group: str, n: int, retries: int = 4) -> list[str]:
    prompt = (
        f"Группа ограничений: {group}.\n"
        f"Жалоба: {text}\n\n"
        f"Сгенерируй {n} разных перефразировок этой жалобы. "
        f'Верни JSON-объект вида {{"paraphrases": ["...", "..."]}}.'
    )
    for attempt in range(retries):
        try:
            resp = client.chat.complete(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return json.loads(content).get("paraphrases", [])[:n]
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} через {wait}s ({exc})")
            time.sleep(wait)
    return []


def main(target_per_class: int = 400, per_call: int = 5, delay: float = 1.0) -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    rows = [r for r in load_split("train") if not is_dropped(r["code"])]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[group_of(r["code"])].append(r)

    out_rows = list(rows)
    for group, items in sorted(by_class.items()):
        need = target_per_class - len(items)
        if need <= 0:
            print(f"{group:18s} {len(items):4d} -> без аугментации")
            continue
        made, i = 0, 0
        while made < need:
            src = items[i % len(items)]
            batch = paraphrase(client, src["symptoms"], group, min(per_call, need - made))
            for para in batch:
                out_rows.append(
                    {"idx": f"{src['idx']}_aug{made}", "symptoms": para, "code": src["code"]}
                )
                made += 1
            i += 1
            time.sleep(delay)
        print(f"{group:18s} {len(items):4d} -> +{need} = {target_per_class}")

    path = DATA_DIR / "train_aug_v1.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nсохранено: {path}  (всего {len(out_rows)} строк)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=400)
    parser.add_argument("--per-call", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    main(args.target, args.per_call, args.delay)
