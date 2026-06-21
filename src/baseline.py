"""Baseline: tf-idf + логистическая регрессия.

Тексты лемматизируются (pymorphy3): для русского флективного языка это схлопывает
словоформы (болит/болела/боли -> болеть) и уменьшает разреженность tf-idf.
"""
import argparse
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_xy
from .evaluate import (
    pr_auc,
    print_dangerous_recall,
    print_report,
    save_confusion,
    save_metrics,
    save_pr_curves,
    save_report,
    scores,
)
from .mapping import GROUPS

NAME = "baseline"
MODELS = Path(__file__).resolve().parent.parent / "models"

_TOKEN = re.compile(r"[а-яёa-z]+")
_morph = None
_lemma_cache: dict[str, str] = {}


def lemmatize(text: str) -> str:
    """Нижний регистр + лемматизация токенов (с кэшем). Используется как preprocessor tf-idf."""
    global _morph
    if _morph is None:
        import pymorphy3

        _morph = pymorphy3.MorphAnalyzer()
    out = []
    for tok in _TOKEN.findall(text.lower()):
        lem = _lemma_cache.get(tok)
        if lem is None:
            lem = _morph.parse(tok)[0].normal_form
            _lemma_cache[tok] = lem
        out.append(lem)
    return " ".join(out)


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=lemmatize, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
    ])


def main(eval_split: str = "dev") -> None:
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)

    model = build_model().fit(x_train, y_train)
    pred = model.predict(x_eval)
    proba = model.predict_proba(x_eval)

    values = scores(y_eval, pred)
    values["pr_auc"], _ = pr_auc(y_eval, proba, model.classes_)
    print(f"{NAME} on {eval_split}: {values}")
    print_report(y_eval, pred, labels=list(GROUPS))
    print_dangerous_recall(y_eval, pred)

    save_confusion(y_eval, pred, list(GROUPS), NAME, eval_split)
    save_pr_curves(y_eval, proba, model.classes_, NAME, eval_split)
    save_report(NAME, eval_split, y_eval, pred, list(GROUPS))
    save_metrics(NAME, eval_split, values)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{NAME}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    main(parser.parse_args().split)
