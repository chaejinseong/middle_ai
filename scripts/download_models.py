"""위험 감지·대응 기능에 필요한 모델 일괄 다운로드

사용:
  python scripts/download_models.py              # 기본(필수) 모델만  — 약 2GB
  python scripts/download_models.py --large      # 대형 가드레일(Qwen3Guard 4B, Kanana Safeguard 8B)까지 — +약 25GB
  python scripts/download_models.py --stt        # Whisper(STT)까지 (requirements-stt 설치 필요)
  python scripts/download_models.py --list       # 목록과 상태만 보기

모델은 Hugging Face 공용 캐시(~/.cache/huggingface/hub)에 저장되어, 코드에서 이름으로 바로 불러집니다.
"""
import argparse, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, scan_cache_dir

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

#          그룹      HF 모델 ID                          기능                      대략 크기
MODELS = [
    ("base",  "beomi/KcELECTRA-base-v2022",      "F6 위험도 분류기 (학습)",       "0.5GB"),
    ("base",  "kiyuyeon/pii-ner-kcelectra",      "F2 개인정보 마스킹",            "0.5GB"),
    ("base",  "Qwen/Qwen3Guard-Gen-0.6B",        "F4 안전 필터 (경량)",           "1.2GB"),
    ("large", "Qwen/Qwen3Guard-Gen-4B",          "F4 안전 필터 (비교)",           "8GB"),
    ("large", "kakaocorp/kanana-safeguard-8b",   "F4 안전 필터 (1순위, 한국어)",   "16GB"),
]
WHISPER_SIZE = "small"   # tiny/base/small/medium/large-v3 — small이 맥에서 속도·정확도 균형
IGNORE = ["*.onnx", "*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_model*", "*.tflite", "onnx/*"]


def cached_repos():
    try:
        return {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true")
    ap.add_argument("--stt", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    have = cached_repos()
    print(f"{'상태':4} {'모델':34} {'기능':26} 크기")
    for g, rid, role, size in MODELS:
        mark = "✅" if rid in have else "⬜"
        print(f"{mark:3} {rid:34} {role:24} {size}{'  (--large)' if g == 'large' else ''}")
    print(f"{'':3} whisper-{WHISPER_SIZE:26} {'F1 음성 인식':24} 0.5GB  (--stt)")
    print("※ Solar(F8 브리핑)는 API라 다운로드 없음 — .env 에 UPSTAGE_API_KEY 입력")
    if a.list:
        return

    targets = [m for m in MODELS if m[0] == "base" or a.large]
    free = shutil.disk_usage(Path.home()).free / 1e9
    print(f"\n디스크 여유 공간: {free:.0f}GB")
    if a.large and free < 35:
        print("⚠ 대형 모델(약 25GB)을 받기엔 여유 공간이 부족할 수 있어요.")

    token = os.getenv("HF_TOKEN") or None
    fails = []
    for g, rid, role, size in targets:
        if rid in have:
            print(f"⏭  이미 있음: {rid}")
            continue
        print(f"\n⬇  {rid}  ({role}, {size})")
        try:
            snapshot_download(rid, ignore_patterns=IGNORE, token=token)
            print(f"✅ 완료: {rid}")
        except Exception as e:
            fails.append(rid)
            print(f"❌ 실패: {rid} — {type(e).__name__}: {e}")

    if a.stt:
        try:
            import whisper
            print(f"\n⬇  whisper-{WHISPER_SIZE}")
            whisper.load_model(WHISPER_SIZE)
            print("✅ 완료: whisper")
        except ImportError:
            print("❌ whisper 미설치 → ./setup.sh --stt 먼저 실행")
            fails.append("whisper")

    print("\n" + ("🎉 모든 모델 준비 완료" if not fails else f"⚠ 실패: {fails} — 네트워크/HF_TOKEN 확인 후 다시 실행"))
    print("현재 상태 확인: python scripts/download_models.py --list")


if __name__ == "__main__":
    main()
