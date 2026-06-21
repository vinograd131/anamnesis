"""Поведенческие тесты в духе CheckList (Ribeiro et al., ACL 2020).

Три типа тестов:
  INV  — перефраз (см. robustness.py): смысл сохранён → предсказание не должно меняться.
  DIR  — отрицание (negate.py): главный симптом отрицается → уверенность в исходном
         классе должна падать. Модель-словарь (tf-idf) видит то же слово и держит
         уверенность; семантическая модель понимает отрицание.
  MFT  — минимальная функциональность: однозначные однострочные жалобы с известной
         группой → проверяем базовый навык.

Логика теста одна для всех моделей: на вход подаётся predict_proba(texts) -> матрица
вероятностей и список классов в порядке колонок.
"""
import json
from pathlib import Path

import numpy as np

from .data import load_split, load_xy
from .mapping import group_of, is_dropped

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


def _true_labels(split: str) -> list[str]:
    return [group_of(r["code"]) for r in load_split(split) if not is_dropped(r["code"])]


def run_negation(predict_proba, classes, model_name: str,
                 orig_split="test", neg_split="test_neg") -> dict:
    """DIR: симптом отрицается → уверенность в исходном классе должна падать.

    Метрики: средняя P(истинный класс) до/после, среднее падение, доля примеров с
    падением, и flip-rate (топ-1 предсказание сменилось). Чем сильнее падение и выше
    flip-rate, тем лучше модель реагирует на отрицание.
    """
    classes = list(classes)
    col = {c: i for i, c in enumerate(classes)}

    x_orig, _ = load_xy(orig_split)
    x_neg, _ = load_xy(neg_split)
    y_true = _true_labels(orig_split)
    assert len(x_orig) == len(x_neg) == len(y_true), "сплиты должны совпадать по длине/порядку"

    p_orig = np.asarray(predict_proba(x_orig))
    p_neg = np.asarray(predict_proba(x_neg))

    rows = np.arange(len(y_true))
    true_col = np.array([col[y] for y in y_true])
    conf_orig = p_orig[rows, true_col]
    conf_neg = p_neg[rows, true_col]
    drop = conf_orig - conf_neg

    top1_orig = p_orig.argmax(1)
    top1_neg = p_neg.argmax(1)
    flip = top1_orig != top1_neg

    values = {
        "n": len(y_true),
        "mean_p_true_orig": round(float(conf_orig.mean()), 4),
        "mean_p_true_neg": round(float(conf_neg.mean()), 4),
        "mean_drop": round(float(drop.mean()), 4),
        "pct_dropped": round(float((drop > 0).mean()), 4),
        "flip_rate": round(float(flip.mean()), 4),
    }
    print(f"[DIR/negation] {model_name}:")
    print(f"    P(истинный класс): {values['mean_p_true_orig']} -> {values['mean_p_true_neg']}"
          f"  (падение {values['mean_drop']:+.3f})")
    print(f"    доля с падением уверенности: {values['pct_dropped']:.3f}")
    print(f"    flip-rate (сменился топ-1):  {values['flip_rate']:.3f}")
    _save("negation", model_name, values)
    return values


def baseline_predictor():
    """tf-idf + LogReg, обученный на train. Возвращает (predict_proba, classes)."""
    from .baseline import build_model

    x_train, y_train = load_xy("train")
    model = build_model().fit(x_train, y_train)
    return (lambda texts: model.predict_proba(texts)), list(model.classes_)


def main() -> None:
    """Локальный прогон baseline: MFT всегда, negation — если есть test_neg."""
    predict_proba, classes = baseline_predictor()
    run_mft(predict_proba, classes, "baseline")
    if (Path(__file__).resolve().parent.parent / "data" / "test_neg_v1.jsonl").exists():
        run_negation(predict_proba, classes, "baseline")
    else:
        print("\ndata/test_neg_v1.jsonl не найден — negation пропущен (сгенерируйте negate.py)")


if __name__ == "__main__":
    main()
