"""Файнтюн RuBioRoBERTa на аугментированном train (train_aug). Дефолты — лучшие параметры Optuna."""
import argparse

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

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
from .transformer_ft import (
    ID2LABEL,
    LABEL2ID,
    MODEL_ID,
    MODELS,
    SEED,
    TextDataset,
    WeightedTrainer,
    _predict_logits,
    _softmax,
    build_model,
    compute_metrics,
    encode,
)

NAME = "rubioroberta_ft_aug"
TRAIN_SPLIT = "train_aug"


def prepare(eval_split, tokenizer, max_length=256, train_split=TRAIN_SPLIT):
    x_train, y_train = load_xy(train_split)
    x_eval, y_eval = load_xy(eval_split)
    y_train = [LABEL2ID[y] for y in y_train]
    y_eval = [LABEL2ID[y] for y in y_eval]
    train_ds = TextDataset(encode(tokenizer, x_train, max_length), y_train)
    eval_ds = TextDataset(encode(tokenizer, x_eval, max_length), y_eval)
    weights = compute_class_weight("balanced", classes=np.arange(len(GROUPS)), y=y_train)
    return train_ds, eval_ds, torch.tensor(weights, dtype=torch.float)


def train(
    eval_split="dev",
    use_lora=True,
    learning_rate=0.0004939977445894489,
    epochs=8,
    batch_size=8,
    weight_decay=0.08187605254258956,
    warmup_ratio=0.02346078831477152,
    patience=2,
    max_length=256,
    train_split=TRAIN_SPLIT,
    name=NAME,
    output_dir="outputs/ft_aug",
    save=True,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds, eval_ds, class_weights = prepare(eval_split, tokenizer, max_length, train_split)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=build_model(use_lora),
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    values = {
        "accuracy": round(metrics["eval_accuracy"], 4),
        "macro_f1": round(metrics["eval_macro_f1"], 4),
    }
    pred = trainer.predict(eval_ds)
    proba = _softmax(pred.predictions)
    y_pred = [GROUPS[i] for i in pred.predictions.argmax(-1)]
    y_true = [GROUPS[i] for i in pred.label_ids]
    values["pr_auc"], _ = pr_auc(y_true, proba, list(GROUPS))
    print(f"{name} on {eval_split}: {values}")
    print_report(y_true, y_pred, labels=list(GROUPS))
    print_dangerous_recall(y_true, y_pred)
    save_confusion(y_true, y_pred, list(GROUPS), name, eval_split)
    save_pr_curves(y_true, proba, list(GROUPS), name, eval_split)
    save_report(name, eval_split, y_true, y_pred, list(GROUPS))
    save_metrics(name, eval_split, values)

    if save:
        MODELS.mkdir(exist_ok=True)
        trainer.save_model(str(MODELS / name))
    return values["macro_f1"]


def evaluate_saved(eval_split="test", adapter_dir=None, max_length=256, name=NAME):
    from peft import PeftModel

    adapter_dir = adapter_dir or str(MODELS / name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(GROUPS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_eval, y_eval = load_xy(eval_split)
    logits = _predict_logits(model, tokenizer, x_eval, device, max_length=max_length)
    proba = _softmax(logits)
    y_pred = [GROUPS[i] for i in logits.argmax(-1)]

    values = scores(y_eval, y_pred)
    values["pr_auc"], _ = pr_auc(y_eval, proba, list(GROUPS))
    print(f"{name} on {eval_split}: {values}")
    print_report(y_eval, y_pred, labels=list(GROUPS))
    print_dangerous_recall(y_eval, y_pred)
    save_confusion(y_eval, y_pred, list(GROUPS), name, eval_split)
    save_pr_curves(y_eval, proba, list(GROUPS), name, eval_split)
    save_report(name, eval_split, y_eval, y_pred, list(GROUPS))
    save_metrics(name, eval_split, values)
    return values["macro_f1"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--full", action="store_true", help="full fine-tune вместо LoRA")
    args = parser.parse_args()
    train(eval_split=args.split, use_lora=not args.full)
