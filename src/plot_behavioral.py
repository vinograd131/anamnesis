"""График MFT: точность на однозначных жалобах, baseline vs трансформер."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parent.parent / "reports"
MODELS = {"baseline": "tf-idf + LogReg", "rubioroberta_ft": "RuBioRoBERTa файнтюн"}
BLUE, ORANGE = "#3b5bdb", "#e8590c"


def main() -> None:
    data = json.loads((REPORTS / "behavioral.json").read_text(encoding="utf-8"))["mft"]
    names = list(MODELS)
    acc = [data[m]["accuracy"] for m in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(x, acc, 0.5, color=[BLUE, ORANGE])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
    ax.set_xticks(x, [MODELS[m] for m in names])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("точность")
    ax.set_title("MFT — однозначные жалобы (CheckList)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = REPORTS / "behavioral.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
