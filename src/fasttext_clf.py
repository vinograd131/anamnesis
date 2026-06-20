"""fastText (gensim): субсловные эмбеддинги + усреднение документа + LogReg."""
import argparse
import re
from pathlib import Path

import joblib
import numpy as np
from gensim.models import FastText
from sklearn.linear_model import LogisticRegression

from .data import load_xy
from .evaluate import print_report, save_confusion, save_metrics, save_report, scores
from .mapping import GROUPS

NAME = "fasttext"
MODELS = Path(__file__).resolve().parent.parent / "models"
SEED = 42

_TOKEN = re.compile(r"[а-яёa-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def train_embeddings(corpus: list[list[str]]) -> FastText:
    return FastText(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        epochs=20,
        min_n=3,
        max_n=5,
        seed=SEED,
        workers=1,
    )


def doc_vectors(model: FastText, docs: list[list[str]]) -> np.ndarray:
    dim = model.vector_size
    out = np.zeros((len(docs), dim), dtype=np.float32)
    for i, tokens in enumerate(docs):
        vecs = [model.wv[t] for t in tokens if t]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def main(eval_split: str = "dev") -> None:
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)

    train_tokens = [tokenize(t) for t in x_train]
    eval_tokens = [tokenize(t) for t in x_eval]

    ft = train_embeddings(train_tokens)
    x_tr = doc_vectors(ft, train_tokens)
    x_ev = doc_vectors(ft, eval_tokens)

    clf = LogisticRegression(
        max_iter=2000, C=10, class_weight="balanced", random_state=SEED
    ).fit(x_tr, y_train)
    pred = clf.predict(x_ev)

    values = scores(y_eval, pred)
    print(f"{NAME} on {eval_split}: {values}")
    print_report(y_eval, pred, labels=list(GROUPS))

    save_confusion(y_eval, pred, list(GROUPS), NAME, eval_split)
    save_report(NAME, eval_split, y_eval, pred, list(GROUPS))
    save_metrics(NAME, eval_split, values)

    MODELS.mkdir(exist_ok=True)
    ft.save(str(MODELS / f"{NAME}.model"))
    joblib.dump(clf, MODELS / f"{NAME}_clf.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    main(parser.parse_args().split)
