"""RuBioRoBERTa с файнтюном: LoRA + взвешенный loss под дисбаланс классов."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .data import load_xy
from .evaluate import save_metrics
from .mapping import GROUPS

NAME = "rubioroberta_ft"
MODEL_ID = "alexyalunin/RuBioRoBERTa"
MODELS = Path(__file__).resolve().parent.parent / "models"
SEED = 42

LABEL2ID = {g: i for i, g in enumerate(GROUPS)}
ID2LABEL = {i: g for g, i in LABEL2ID.items()}


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device)
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial
        self.epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import optuna

        self.trial.report(metrics["eval_macro_f1"], step=self.epoch)
        self.epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def compute_metrics(pred):
    y_pred = pred.predictions.argmax(-1)
    y_true = pred.label_ids
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def encode(tokenizer, texts, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def build_model(use_lora: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(GROUPS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)
    return model


def prepare(eval_split, tokenizer, max_length=128):
    x_train, y_train = load_xy("train")
    x_eval, y_eval = load_xy(eval_split)
    y_train = [LABEL2ID[y] for y in y_train]
    y_eval = [LABEL2ID[y] for y in y_eval]
    train_ds = TextDataset(encode(tokenizer, x_train, max_length), y_train)
    eval_ds = TextDataset(encode(tokenizer, x_eval, max_length), y_eval)
    weights = compute_class_weight("balanced", classes=np.arange(len(GROUPS)), y=y_train)
    return train_ds, eval_ds, torch.tensor(weights, dtype=torch.float)


def train_once(
    eval_split="dev",
    use_lora=True,
    learning_rate=2e-4,
    epochs=10,
    batch_size=8,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=2,
    output_dir="outputs/ft",
    save=False,
    trial=None,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_ds, eval_ds, class_weights = prepare(eval_split, tokenizer)

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
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    if trial is not None:
        callbacks.append(OptunaPruningCallback(trial))

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=build_model(use_lora),
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    metrics = trainer.evaluate()
    values = {
        "accuracy": round(metrics["eval_accuracy"], 4),
        "macro_f1": round(metrics["eval_macro_f1"], 4),
    }
    print(f"{NAME} on {eval_split}: {values}")
    save_metrics(NAME, eval_split, values)
    if save:
        MODELS.mkdir(exist_ok=True)
        trainer.save_model(str(MODELS / NAME))
    return values["macro_f1"]


def run_optuna(n_trials=20, eval_split="dev", use_lora=True):
    import optuna

    def objective(trial):
        return train_once(
            eval_split=eval_split,
            use_lora=use_lora,
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            epochs=trial.suggest_int("epochs", 4, 10),
            batch_size=trial.suggest_categorical("batch_size", [8, 16]),
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
            output_dir=f"outputs/trial_{trial.number}",
            trial=trial,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    print("best macro_f1:", study.best_value)
    print("best params:", study.best_params)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--full", action="store_true", help="full fine-tune вместо LoRA")
    parser.add_argument("--optuna", type=int, default=0, help="число трайлов Optuna")
    args = parser.parse_args()
    if args.optuna:
        run_optuna(n_trials=args.optuna, eval_split=args.split, use_lora=not args.full)
    else:
        train_once(eval_split=args.split, use_lora=not args.full, save=True)
