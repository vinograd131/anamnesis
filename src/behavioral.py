"""Поведенческие тесты в духе CheckList (Ribeiro et al., ACL 2020).

  INV  — перефраз (см. robustness.py): смысл сохранён → предсказание не должно меняться.
  MFT  — минимальная функциональность: однозначные однострочные жалобы с известной
         группой → проверяем базовый навык.

Логика теста одна для всех моделей: на вход подаётся predict_proba(texts) -> матрица
вероятностей и список классов в порядке колонок.
"""
import json
from pathlib import Path

import numpy as np

from .data import load_xy

REPORTS = Path(__file__).resolve().parent.parent / "reports"
RESULTS_FILE = REPORTS / "behavioral.json"

# MFT: однозначные жалобы бытовым языком, по 4 на каждую из 8 групп.
MFT_CASES: list[tuple[str, str]] = [
    ("болит поясница после долгого сидения", "ПОЗВОНОЧНИК"),
    ("прострел в спине, тяжело разогнуться", "ПОЗВОНОЧНИК"),
    ("ноющая боль в шее и между лопаток", "ПОЗВОНОЧНИК"),
    ("скованность в пояснице по утрам", "ПОЗВОНОЧНИК"),
    ("опухло и болит колено", "СУСТАВЫ"),
    ("ноют суставы пальцев рук", "СУСТАВЫ"),
    ("боль в плече, не могу поднять руку", "СУСТАВЫ"),
    ("хрустит и болит тазобедренный сустав", "СУСТАВЫ"),
    ("давит в груди и поднимается давление", "КАРДИО+НЕВРО"),
    ("сильное сердцебиение и перебои в сердце", "КАРДИО+НЕВРО"),
    ("частые головные боли и головокружение", "КАРДИО+НЕВРО"),
    ("немеют пальцы на руке", "КАРДИО+НЕВРО"),
    ("кашель и заложенность носа", "ДЫХАТЕЛЬНЫЕ"),
    ("одышка и хрипы при дыхании", "ДЫХАТЕЛЬНЫЕ"),
    ("болит горло и насморк", "ДЫХАТЕЛЬНЫЕ"),
    ("приступы удушья, тяжело дышать", "ДЫХАТЕЛЬНЫЕ"),
    ("набрал лишний вес, постоянная усталость", "ЭНДОКРИН/МЕТАБОЛ"),
    ("повышенный сахар в крови", "ЭНДОКРИН/МЕТАБОЛ"),
    ("увеличена щитовидная железа", "ЭНДОКРИН/МЕТАБОЛ"),
    ("потливость, дрожь, похудел при хорошем аппетите", "ЭНДОКРИН/МЕТАБОЛ"),
    ("изжога после еды", "ЖКТ/БРЮШНЫЕ"),
    ("боль в животе и тошнота", "ЖКТ/БРЮШНЫЕ"),
    ("тяжесть в правом подреберье после жирного", "ЖКТ/БРЮШНЫЕ"),
    ("вздутие живота и нарушение стула", "ЖКТ/БРЮШНЫЕ"),
    ("жжение и резь при мочеиспускании", "МОЧЕПОЛОВЫЕ/ТАЗ"),
    ("тянущая боль внизу живота перед менструацией", "МОЧЕПОЛОВЫЕ/ТАЗ"),
    ("обильные и нерегулярные менструации", "МОЧЕПОЛОВЫЕ/ТАЗ"),
    ("учащённое болезненное мочеиспускание", "МОЧЕПОЛОВЫЕ/ТАЗ"),
    ("сыпь и зуд на коже", "КОЖА"),
    ("шелушение и покраснение кожи лица", "КОЖА"),
    ("угревая сыпь на лице", "КОЖА"),
    ("крапивница, чешется всё тело", "КОЖА"),
]


def _save(test: str, model_name: str, values: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    data = {}
    if RESULTS_FILE.exists():
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    data.setdefault(test, {})[model_name] = values
    RESULTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mft(predict_proba, classes, model_name: str) -> dict:
    """MFT: точность на однозначных жалобах, общая и по группам."""
    classes = list(classes)
    texts = [t for t, _ in MFT_CASES]
    gold = [g for _, g in MFT_CASES]
    proba = np.asarray(predict_proba(texts))
    pred = [classes[i] for i in proba.argmax(1)]

    correct = sum(p == g for p, g in zip(pred, gold))
    per_group = {}
    for grp in sorted(set(gold)):
        idx = [i for i, g in enumerate(gold) if g == grp]
        per_group[grp] = round(sum(pred[i] == grp for i in idx) / len(idx), 3)

    values = {"accuracy": round(correct / len(gold), 4), "n": len(gold), "per_group": per_group}
    print(f"[MFT] {model_name}: accuracy {values['accuracy']} ({correct}/{len(gold)})")
    for grp, acc in per_group.items():
        flag = "" if acc == 1.0 else "  <-"
        print(f"    {grp:18s} {acc:.3f}{flag}")
    _save("mft", model_name, values)
    return values


def baseline_predictor():
    """tf-idf + LogReg, обученный на train. Возвращает (predict_proba, classes)."""
    from .baseline import build_model

    x_train, y_train = load_xy("train")
    model = build_model().fit(x_train, y_train)
    return (lambda texts: model.predict_proba(texts)), list(model.classes_)


def main() -> None:
    predict_proba, classes = baseline_predictor()
    run_mft(predict_proba, classes, "baseline")


if __name__ == "__main__":
    main()
