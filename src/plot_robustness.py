"""График робастности: macro-F1 на оригинальном test и на перефразированном (свободная речь)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parent.parent / "reports"
MODELS = {"baseline": "tf-idf + LogReg", "rubioroberta_ft": "RuBioRoBERTa\nфайнтюн"}


def main() -> None:
    data = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    names = list(MODELS)
    orig = [data[m]["test"]["macro_f1"] for m in names]
    para = [data[m]["test_para"]["macro_f1"] for m in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - width / 2, orig, width, label="оригинальный test", color="#3b5bdb")
    b2 = ax.bar(x + width / 2, para, width, label="перефраз (свободная речь)", color="#e8590c")

    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
    for m, xi, p in zip(names, x, para):
        drop = data[m]["test_para"]["macro_f1"] - data[m]["test"]["macro_f1"]
        ax.annotate(
            f"{p:.3f}\n({drop:+.2f})",
            xy=(xi + width / 2, p),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#c92a2a",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODELS[m] for m in names])
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Устойчивость к свободной речи (перефразированный test)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = REPORTS / "robustness.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"сохранено: {path}")


if __name__ == "__main__":
    main()
