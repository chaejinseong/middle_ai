"""F6 위험도 분류기(KcELECTRA) 파인튜닝

사용:  python -m src.risk.train_classifier --config configs/risk_classifier.yaml
"""
import argparse, json, shutil
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)

from src.common.io import load_config, resolve, LABELS, LABEL2ID, ID2LABEL
from src.common.device import get_device


def load_df(path):
    df = pd.read_csv(resolve(path))
    df = df.dropna(subset=["text", "label"])
    bad = set(df["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"알 수 없는 라벨: {bad} (허용: {LABELS})")
    df["labels"] = df["label"].map(LABEL2ID)
    return df[["text", "labels"]]


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    y = p.label_ids
    prec, rec, _, _ = precision_recall_fscore_support(
        y, preds, labels=list(range(len(LABELS))), zero_division=0)
    out = {"macro_f1": f1_score(y, preds, average="macro", zero_division=0)}
    out.update({f"precision_{l}": prec[i] for i, l in enumerate(LABELS)})
    out.update({f"recall_{l}": rec[i] for i, l in enumerate(LABELS)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/risk_classifier.yaml")
    ap.add_argument("--model_name", help="config의 model_name 덮어쓰기")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model_name:
        cfg["model_name"] = args.model_name
    set_seed(cfg["seed"])
    print(f"device: {get_device()} | model: {cfg['model_name']}")

    df = load_df(cfg["train_file"])
    counts = df["labels"].value_counts()
    stratify = df["labels"] if counts.min() >= 2 and len(df) >= 20 else None
    if stratify is None:
        print("⚠ 데이터가 적어 층화 분할 없이 진행합니다 (샘플 데이터 단계).")
    tr, va = train_test_split(df, test_size=cfg["val_ratio"], random_state=cfg["seed"], stratify=stratify)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True)

    def enc(b):
        return tok(b["text"], truncation=True, max_length=cfg["max_length"])

    ds_tr = Dataset.from_pandas(tr, preserve_index=False).map(enc, batched=True)
    ds_va = Dataset.from_pandas(va, preserve_index=False).map(enc, batched=True)

    out_dir = resolve(cfg["output_dir"])
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "val_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)   # 중간 저장본 삭제(용량 절약)
    print(f"✅ 저장: {out_dir}")


if __name__ == "__main__":
    main()
