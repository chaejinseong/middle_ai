"""가상환경 점검: 패키지 버전, 사용 장치(MPS/CUDA/CPU), 간단한 연산, HF 접속"""
import importlib, platform, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print(f"Python  : {sys.version.split()[0]}  ({sys.executable})")
print(f"OS      : {platform.platform()}")
ok = True
for pkg in ["torch", "transformers", "datasets", "accelerate", "sklearn", "pandas",
            "kiwipiepy", "yaml", "openai", "dotenv"]:
    try:
        m = importlib.import_module(pkg)
        print(f"  ✅ {pkg:13} {getattr(m, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  ❌ {pkg:13} {e}")

import torch
from src.common.device import get_device
dev = get_device()
x = torch.randn(256, 256, device=dev)
print(f"장치    : {dev}  (행렬곱 테스트 OK: {float((x @ x).sum()):.1f})")
if dev == "cpu":
    print("  ⚠ GPU 가속 없음 — 작은 모델 학습은 가능하지만 느릴 수 있어요.")

from kiwipiepy import Kiwi
print("형태소  :", [t.form for t in Kiwi().tokenize("요즘 잠을 못 자요")])

from src.risk.rules import RuleScorer
print("규칙    :", RuleScorer().score("요즘 진짜 죽고 싶어").risk_level, "(예상: high)")

try:
    from huggingface_hub import HfApi
    HfApi().model_info("beomi/KcELECTRA-base-v2022")
    print("HF 접속 : OK")
except Exception as e:
    print(f"HF 접속 : 실패 ({type(e).__name__}) — 네트워크 확인")

print("\n결과    :", "모든 점검 통과 ✅" if ok else "일부 패키지 문제 ❌")
