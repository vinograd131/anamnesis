"""RuBioRoBERTa без файнтюна: замороженные эмбеддинги (mean-pooling) + MLP."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer

from .data import load_xy
from .evaluate import print_report, save_confusion, save_metrics, scores
from .mapping import GROUPS

NAME = "rubioroberta_frozen_mlp"
MODEL_ID = "alexyalunin/RuBioRoBERTa"
MODELS = Path(__file__).resolve().parent.parent / "models"
SEED = 42


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(texts, tokenizer, model, dev, batch_size=32, max_length=256) -> np.ndarray:
    model.eval().to(dev)
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(dev)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks)


def cached_embed(split, texts, tokenizer, model, dev) -> np.ndarray:
    path = MODELS / f"rubioroberta_{split}.npy"
    if path.exists():
        return np.load(path)
    vecs = embed(texts, tokenizer, model, dev)
    MODELS.mkdir(exist_ok=True)
    np.save(path, vecs)
    return vecs


def main(eval_split: str = "dev") -> None:
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    dev = device()

    x_tr = cached_embed("train", x_train, tokenizer, model, dev)
    x_ev = cached_embed(eval_split, x_eval, tokenizer, model, dev)

    label2id = {g: i for i, g in enumerate(GROUPS)}
    y_tr = [label2id[y] for y in y_train]

    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        max_iter=300,
        early_stopping=True,
        random_state=SEED,
    ).fit(x_tr, y_tr)
    pred = [GROUPS[i] for i in clf.predict(x_ev)]

    values = scores(y_eval, pred)
    print(f"{NAME} on {eval_split}: {values}")
    print_report(y_eval, pred, labels=list(GROUPS))

    save_confusion(y_eval, pred, list(GROUPS), NAME, eval_split)
    save_metrics(NAME, eval_split, values)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS / f"{NAME}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    main(parser.parse_args().split)
