"""Метрики и артефакты оценки, общие для всех моделей."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"
METRICS_FILE = REPORTS / "metrics.json"


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def print_report(y_true, y_pred, labels=None) -> None:
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))


def save_report(name: str, split: str, y_true, y_pred, labels) -> Path:
    REPORTS.mkdir(exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    path = REPORTS / f"report_{name}_{split}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_confusion(y_true, y_pred, labels, name: str, split: str = "dev") -> Path:
    REPORTS.mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format=".2f")
    ax.set_title(f"{name} — confusion ({split})")
    fig.tight_layout()
    path = REPORTS / f"confusion_{name}_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_metrics(name: str, split: str, values: dict[str, float]) -> None:
    REPORTS.mkdir(exist_ok=True)
    data = {}
    if METRICS_FILE.exists():
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    data.setdefault(name, {})[split] = values
    METRICS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
