"""F5 추세 악화 탐지(trend_flag) + 합성 환자 궤적 생성/평가

사용:
  python -m src.trend.trend_flag --simulate 60      # 합성 환자 60명 생성 → 탐지 성능 출력
"""
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

from src.common.io import resolve


def trend_flag(scores, window=7, rise=12.0, slope_window=14, slope_th=1.0, z_th=2.5,
               baseline_days=14, persist=3):
    """일별 위험/우울 점수(0~100) 리스트 → 날짜별 flag 리스트

    다음 중 하나가 persist일 연속 유지되면 flag (노이즈로 인한 오경보 억제):
      ① 최근 window일 평균이 직전 window일 평균보다 rise 이상 상승
      ② 최근 slope_window일 선형회귀 기울기 ≥ slope_th (점/일)
      ③ 개인 기준선(처음 baseline_days일) 대비 최근 평균의 z-score ≥ z_th
    결측(미접속)은 보간 — 미접속 자체는 F11 신호로 따로 다룸.
    """
    s = pd.Series(scores, dtype=float).interpolate(limit_direction="both")
    base = s.iloc[:baseline_days]
    mu, sd = base.mean(), max(base.std(), 4.0)
    raw = []
    for t in range(len(s)):
        if t < max(2 * window, slope_window, baseline_days):
            raw.append(False)
            continue
        cur, prev = s.iloc[t - window + 1:t + 1], s.iloc[t - 2 * window + 1:t - window + 1]
        c1 = cur.mean() - prev.mean() >= rise
        sw = s.iloc[t - slope_window + 1:t + 1].values
        c2 = np.polyfit(np.arange(slope_window), sw, 1)[0] >= slope_th
        c3 = (cur.mean() - mu) / sd >= z_th
        raw.append(bool(c1 or c2 or c3))
    flags, run = [], 0
    for r in raw:
        run = run + 1 if r else 0
        flags.append(run >= persist)
    return flags


def simulate(n=60, days=56, seed=0):
    """안정형/점진악화형/급변형/회복형/미접속동반형 합성 궤적"""
    rng = np.random.default_rng(seed)
    types = ["stable", "gradual", "sudden", "recovering", "gradual_dropout"]
    rows = []
    for pid in range(n):
        kind = types[pid % len(types)]
        base = rng.uniform(15, 35)
        onset = None
        x = np.full(days, base)
        if kind in ("gradual", "gradual_dropout"):
            onset = int(rng.integers(18, 35))
            x[onset:] += np.linspace(0, rng.uniform(25, 45), days - onset)
        elif kind == "sudden":
            onset = int(rng.integers(18, 45))
            x[onset:] += rng.uniform(30, 50)
        elif kind == "recovering":
            x = np.linspace(base + 30, base, days)
        x = np.clip(x + rng.normal(0, 4, days), 0, 100)
        if kind == "gradual_dropout":            # 악화 후 미접속(기록 결측)
            x = x.astype(float)
            x[onset + 7:] = np.nan
        for d in range(days):
            rows.append({"patient_id": pid, "type": kind, "day": d, "score": x[d], "onset": onset})
    return pd.DataFrame(rows)


def evaluate(df, **kw):
    out = []
    for pid, g in df.groupby("patient_id"):
        flags = trend_flag(g["score"].tolist(), **kw)
        onset = g["onset"].iloc[0]
        first = next((d for d, f in enumerate(flags) if f), None)
        if pd.isna(onset):
            out.append({"patient_id": pid, "type": g["type"].iloc[0], "detected": first is not None,
                        "delay_days": None, "false_alarm": first is not None})
        else:
            hit = next((d for d, f in enumerate(flags) if f and d >= onset), None)
            early = any(flags[:int(onset)])
            out.append({"patient_id": pid, "type": g["type"].iloc[0], "detected": hit is not None,
                        "delay_days": None if hit is None else hit - int(onset), "false_alarm": early})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int, default=60)
    ap.add_argument("--input", help="실제 CSV (patient_id, day, score[, onset])")
    args = ap.parse_args()
    df = pd.read_csv(resolve(args.input)) if args.input else simulate(args.simulate)
    if "type" not in df:
        df["type"] = "real"
    if "onset" not in df:
        df["onset"] = np.nan
    res = evaluate(df)
    summary = res.groupby("type").agg(
        patients=("patient_id", "count"),
        detection_rate=("detected", "mean"),
        mean_delay_days=("delay_days", "mean"),
        false_alarm_rate=("false_alarm", "mean"))
    print(summary.round(2).to_string())
    out = resolve("outputs") / f"trend_eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trajectories.csv", index=False)
    res.to_csv(out / "per_patient.csv", index=False)
    summary.to_csv(out / "summary.csv")
    print(f"📁 결과 저장: {out}")


if __name__ == "__main__":
    main()
