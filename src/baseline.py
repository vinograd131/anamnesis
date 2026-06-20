"""Baseline: tf-idf + логистическая регрессия."""
import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_xy
from .evaluate import print_report, save_confusion, save_metrics, save_report, scores
from .mapping import GROUPS

NAME = "baseline"
MODELS = Path(__file__).resolve().parent.parent / "models"


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
    ])


def main(eval_split: str = "dev") -> None:
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)

    model = build_model().fit(x_train, y_train)
    pred = model.predict(x_eval)

    values = scores(y_eval, pred)
    print(f"{NAME} on {eval_split}: {values}")
    print_report(y_eval, pred, labels=list(GROUPS))

    save_confusion(y_eval, pred, list(GROUPS), NAME, eval_split)
    save_report(NAME, eval_split, y_eval, pred, list(GROUPS))
    save_metrics(NAME, eval_split, values)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{NAME}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    main(parser.parse_args().split)
