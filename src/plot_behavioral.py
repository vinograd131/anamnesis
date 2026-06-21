"""Графики поведенческих тестов: MFT (точность) и DIR/отрицание (падение уверенности)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parent.parent / "reports"
MODELS = {"baseline": "tf-idf\n+ LogReg", "rubioroberta_ft": "RuBioRoBERTa\nфайнтюн"}
BLUE, ORANGE = "#3b5bdb", "#e8590c"


def main() -> None:
    data = json.loads((REPORTS / "behavioral.json").read_text(encoding="utf-8"))
    names = list(MODELS)
    labels = [MODELS[m] for m in names]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    mft = [data["mft"][m]["accuracy"] for m in names]
    b1 = ax1.bar(x, mft, 0.5, color=[BLUE, ORANGE])
    ax1.bar_label(b1, fmt="%.3f", padding=3)
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("точность")
    ax1.set_title("MFT — однозначные жалобы")
    ax1.grid(axis="y", alpha=0.3)

    drop = [data["negation"][m]["mean_drop"] for m in names]
    b2 = ax2.bar(x, drop, 0.5, color=[BLUE, ORANGE])
    ax2.bar_label(b2, fmt="%.3f", padding=3)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("падение P(истинный класс)")
    ax2.set_title("DIR — реакция на отрицание симптома\n(больше = лучше)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Поведенческие тесты (CheckList)", fontsize=13)
    fig.tight_layout()
    path = REPORTS / "behavioral.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
