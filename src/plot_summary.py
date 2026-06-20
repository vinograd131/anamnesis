"""Сводный график сравнения моделей на test: macro-F1 и PR AUC."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parent.parent / "reports"

ORDER = ["rubioroberta_frozen_mlp", "fasttext", "catboost", "baseline", "rubioroberta_ft"]
LABELS = {
    "baseline": "tf-idf\n+ LogReg",
    "fasttext": "fastText\n+ LogReg",
    "catboost": "fastText\n+ CatBoost",
    "rubioroberta_frozen_mlp": "RuBioRoBERTa\nfrozen + MLP",
    "rubioroberta_ft": "RuBioRoBERTa\nфайнтюн",
}


def main(split: str = "test") -> None:
    data = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    names = [m for m in ORDER if m in data and split in data[m]]
    f1 = [data[m][split]["macro_f1"] for m in names]
    ap = [data[m][split].get("pr_auc") for m in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_f1 = ax.bar(x - width / 2, f1, width, label="macro-F1", color="#3b5bdb")
    bars_ap = ax.bar(x + width / 2, ap, width, label="PR AUC", color="#2b8a3e")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in names], fontsize=9)
    ax.set_ylim(0.78, 0.95)
    ax.set_ylabel("score")
    ax.set_title(f"Сравнение моделей ({split})")
    ax.legend(loc="lower right")
    ax.bar_label(bars_f1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars_ap, fmt="%.3f", fontsize=8, padding=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = REPORTS / f"summary_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    main(parser.parse_args().split)
