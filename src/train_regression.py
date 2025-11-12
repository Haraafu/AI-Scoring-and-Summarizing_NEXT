# src/train_regression.py
import os, numpy as np, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")
OUT_DIR    = os.getenv("MODEL_DIR",  "models/roberta-risk")
MAX_LEN    = 384

def load_ds(csv_path):
    df = pd.read_csv(csv_path)
    assert {"text","score"}.issubset(df.columns), "CSV must have columns: text,score"
    # skala 0..1 biar stabil saat training
    df["label01"] = df["score"].clip(0, 100) / 100.0
    return Dataset.from_pandas(df[["text","label01"]])

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    train_ds = load_ds("data/train.csv").map(tok_fn, batched=True)
    valid_ds = load_ds("data/valid.csv").map(tok_fn, batched=True)

    # >>> penting: rename ke 'labels' agar Trainer menghitung loss
    train_ds = train_ds.rename_column("label01", "labels")
    valid_ds = valid_ds.rename_column("label01", "labels")

    # set format: pastikan 'labels' dibawa dan bertipe float32
    def to_float32(ex):
        ex["labels"] = np.asarray(ex["labels"], dtype="float32")
        return ex
    train_ds = train_ds.map(to_float32)
    valid_ds = valid_ds.map(to_float32)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    valid_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",  # minta loss regresi (MSE) otomatis
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        preds_100  = np.clip(preds,  0, 1) * 100.0
        labels_100 = np.clip(labels, 0, 1) * 100.0
        mae  = float(np.mean(np.abs(preds_100 - labels_100)))
        rmse = float(np.sqrt(np.mean((preds_100 - labels_100)**2)))
        return {"mae": mae, "rmse": rmse}

    # Catatan: di 4.57, evaluation_strategy deprecated → pakai eval_strategy (opsional).
    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        do_eval=True,            # kita tetap evaluasi (dipanggil manual/otomatis tergantung versi)
        report_to=[],            # matikan wandb/tb default
        logging_steps=50,
        save_total_limit=2,
        # Jika ingin per-epoch dan versi kamu mendukung:
        # eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        # load_best_model_at_end=True, metric_for_best_model="rmse", greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=valid_ds)
    print("Validation metrics:", metrics)

    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

if __name__ == "__main__":
    main()
