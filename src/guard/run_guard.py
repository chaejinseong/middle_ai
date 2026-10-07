"""F4 안전 필터 게이트 A/B — 기성 가드레일 모델 테스트

- kanana   : kakaocorp/kanana-safeguard-8b  (<SAFE> / <UNSAFE-S1~S7>, S6=자살·자해)
- qwen0.6b : Qwen/Qwen3Guard-Gen-0.6B       (Safe / Controversial / Unsafe)
- qwen4b   : Qwen/Qwen3Guard-Gen-4B

입력 CSV 컬럼: user, response, expected
  - response 가 비어 있으면 '사용자 입력(게이트 A)'만 검사
  - expected: safe / unsafe  (패턴 정보는 pattern 컬럼에 자유 기재)

사용:  python -m src.guard.run_guard --model qwen0.6b --input data/custom/gate_b_responses.csv
"""
import argparse, re, time
from datetime import datetime
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.common.io import resolve
from src.common.device import get_device, get_dtype

MODELS = {
    "kanana": "kakaocorp/kanana-safeguard-8b",
    "qwen0.6b": "Qwen/Qwen3Guard-Gen-0.6B",
    "qwen4b": "Qwen/Qwen3Guard-Gen-4B",
}


def build_messages(user, response):
    msgs = [{"role": "user", "content": user}]
    if isinstance(response, str) and response.strip():
        msgs.append({"role": "assistant", "content": response})
    return msgs


def parse(kind, text):
    if kind == "kanana":
        m = re.search(r"<(SAFE|UNSAFE-S\d)>", text)
        label = m.group(1) if m else text.strip()
        return ("unsafe" if label.startswith("UNSAFE") else "safe"), label
    sev = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", text)
    cat = re.search(r"Categories:\s*(.+)", text)
    s = sev.group(1) if sev else "?"
    verdict = "unsafe" if s in ("Unsafe", "Controversial") else "safe"
    return verdict, f"{s} | {cat.group(1).strip() if cat else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="qwen0.6b")
    ap.add_argument("--input", default="data/custom/gate_b_responses.csv")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="4bit 양자화 로드 (CUDA 전용, Colab T4에서 8B 모델 실행 시 필요)")
    ap.add_argument("--controversial_as_safe", action="store_true",
                    help="Qwen3Guard의 Controversial을 safe로 취급(과차단 비교용)")
    args = ap.parse_args()

    dev = get_device()
    name = MODELS[args.model]
    print(f"device: {dev} | model: {name}  (처음 실행 시 다운로드에 시간이 걸립니다)")
    tok = AutoTokenizer.from_pretrained(name)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=get_dtype(dev)).to(dev).eval()

    df = pd.read_csv(resolve(args.input)).fillna("")
    rows = []
    for _, r in df.iterrows():
        ids = tok.apply_chat_template(build_messages(r["user"], r["response"]),
                                      add_generation_prompt=(args.model != "kanana"),
                                      return_tensors="pt")
        if not torch.is_tensor(ids):          # transformers 버전에 따라 dict로 반환됨
            ids = ids["input_ids"]
        ids = ids.to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=1 if args.model == "kanana" else 64,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        ms = (time.time() - t0) * 1000
        raw = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
        verdict, label = parse(args.model, raw)
        if args.controversial_as_safe and label.startswith("Controversial"):
            verdict = "safe"
        rows.append({**r.to_dict(), "pred": verdict, "raw_label": label, "latency_ms": round(ms)})
        print(f"[{verdict:6}] {label:28} {ms:6.0f}ms | {str(r['response'] or r['user'])[:40]}")

    res = pd.DataFrame(rows)
    if "expected" in res and (res["expected"] != "").any():
        unsafe = res[res.expected == "unsafe"]
        safe = res[res.expected == "safe"]
        print("\n===== 요약 =====")
        if len(unsafe):
            print(f"위험 응답 차단율(재현율): {(unsafe.pred == 'unsafe').mean():.1%}  ({len(unsafe)}건)")
            if "pattern" in res:
                print(unsafe.assign(hit=unsafe.pred == "unsafe").groupby("pattern")["hit"].mean().map("{:.0%}".format))
        if len(safe):
            print(f"정상 응답 과차단율: {(safe.pred == 'unsafe').mean():.1%}  ({len(safe)}건)")
        print(f"평균 지연: {res.latency_ms.mean():.0f}ms")
    out = resolve("outputs") / f"guard_{args.model}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"📁 결과 저장: {out}")


if __name__ == "__main__":
    main()
