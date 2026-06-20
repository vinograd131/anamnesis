"""Метрики и артефакты оценки, общие для всех моделей."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.preprocessing import label_binarize

REPORTS = Path(__file__).resolve().parent.parent / "reports"
METRICS_FILE = REPORTS / "metrics.json"

# Классы с высокой ценой пропуска (для медицины важнее recall именно тут).
DANGEROUS = ("ПОЗВОНОЧНИК", "КАРДИО+НЕВРО", "СУСТАВЫ")


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def print_report(y_true, y_pred, labels=None) -> None:
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))


def pr_auc(y_true, y_proba, proba_labels):
    """PR AUC (average precision) one-vs-rest: macro + по классам.

    proba_labels — порядок колонок y_proba (clf.classes_)."""
    classes = list(proba_labels)
    binarized = label_binarize(y_true, classes=classes)
    per_class = {
        str(lab): round(float(average_precision_score(binarized[:, i], y_proba[:, i])), 4)
        for i, lab in enumerate(classes)
    }
    macro = round(float(np.mean(list(per_class.values()))), 4)
    return macro, per_class


def print_dangerous_recall(y_true, y_pred, groups=DANGEROUS) -> None:
    rec = recall_score(y_true, y_pred, labels=list(groups), average=None, zero_division=0)
    print("recall на опасных классах:")
    for group, value in zip(groups, rec):
        print(f"  {group:18s} {value:.3f}")


def save_pr_curves(y_true, y_proba, proba_labels, name: str, split: str = "test") -> Path:
    """Precision-recall кривые one-vs-rest по всем классам в одной фигуре."""
    REPORTS.mkdir(exist_ok=True)
    classes = list(proba_labels)
    binarized = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    aps = []
    for i, lab in enumerate(classes):
        ap = average_precision_score(binarized[:, i], y_proba[:, i])
        aps.append(ap)
        precision, recall, _ = precision_recall_curve(binarized[:, i], y_proba[:, i])
        ax.plot(recall, precision, label=f"{lab} (AP={ap:.2f})")
    macro = float(np.mean(aps))
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{name} — PR-кривые по классам ({split}), macro AP = {macro:.3f}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = REPORTS / f"pr_curve_{name}_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


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
