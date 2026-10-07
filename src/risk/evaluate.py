"""F6 위험도 평가 — 규칙 / 분류기 / 결합 세 방식을 같은 테스트셋에서 비교

사용:
  python -m src.risk.evaluate --mode rules
  python -m src.risk.evaluate --mode all --test data/custom/hard_negatives.csv
"""
import argparse, json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.common.io import load_config, resolve, LABELS, LABEL2ID
from src.common.device import get_device
from src.risk.rules import RuleScorer

LEVEL_SCORE = {"none": 0, "low": 15, "mid": 45, "high": 80}   # 분류기 확률 → 0~100 환산 기준점


def score_to_level(s, th):
    return "high" if s >= th["high"] else "mid" if s >= th["mid"] else "low" if s >= th["low"] else "none"


def classifier_scores(model_dir, texts, max_length=128):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    w = torch.tensor([LEVEL_SCORE[l] for l in LABELS], dtype=torch.float32)
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**b).logits.float(), -1).cpu()
        out.extend((p * w).sum(-1).tolist())          # 기대 점수
    return out


def report(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("혼동행렬 (행=정답, 열=예측):", LABELS)
    print(cm)
    missed = sum(1 for t, p in zip(y_true, y_pred) if t == "high" and p == "none")
    false_alarm = sum(1 for t, p in zip(y_true, y_pred) if t == "none" and p in ("mid", "high"))
    print(f"고위험→없음 누락: {missed}건 | 위험없음→중/고 오탐: {false_alarm}건")
    rep = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    return {"report": rep, "confusion": cm.tolist(), "missed_high": missed, "false_alarm": false_alarm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/risk_classifier.yaml")
    ap.add_argument("--test", help="테스트 CSV (기본: config의 test_file)")
    ap.add_argument("--mode", choices=["rules", "model", "combined", "all"], default="rules")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(resolve(args.test or cfg["test_file"])).dropna(subset=["text", "label"])
    texts, y = df["text"].tolist(), df["label"].tolist()

    rs = RuleScorer()
    th = rs.th
    results, preds_table = {}, df.copy()
    rule_scores = [rs.score(t).risk_score for t in texts]

    if args.mode in ("rules", "all", "combined"):
        p = [score_to_level(s, th) for s in rule_scores]
        preds_table["rule_score"], preds_table["rule_pred"] = rule_scores, p
        if args.mode != "combined":
            results["rules"] = report("규칙기반", y, p)

    if args.mode in ("model", "all", "combined"):
        mdir = resolve(cfg["output_dir"])
        if not (mdir / "config.json").exists():
            raise SystemExit(f"학습된 모델이 없습니다: {mdir}  → 먼저 train_classifier 실행")
        cls = classifier_scores(str(mdir), texts, cfg["max_length"])
        preds_table["clf_score"] = cls
        if args.mode != "combined":
            p = [score_to_level(s, th) for s in cls]
            preds_table["clf_pred"] = p
            results["model"] = report("분류기(KcELECTRA)", y, p)
        rw = cfg["rule_weight"]
        comb = [rw * r + (1 - rw) * c for r, c in zip(rule_scores, cls)]
        p = [score_to_level(s, th) for s in comb]
        preds_table["combined_score"], preds_table["combined_pred"] = comb, p
        results["combined"] = report(f"결합(규칙 {rw} : 분류기 {1-rw:.1f})", y, p)

    out = resolve("outputs") / f"risk_eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    preds_table.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n📁 결과 저장: {out}")


if __name__ == "__main__":
    main()
