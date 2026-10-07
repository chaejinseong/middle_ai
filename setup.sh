#!/usr/bin/env bash
# middle_project_ai 가상환경 설치 스크립트 (macOS / Linux)
#   ./setup.sh          기본 설치
#   ./setup.sh --stt    STT(Whisper)까지 설치
set -e
cd "$(dirname "$0")"

# 1) Python 3.10~3.12 찾기 (torch·kiwipiepy 호환 범위)
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    ok=$("$c" -c 'import sys; print(int((3,10) <= sys.version_info[:2] <= (3,12)))' 2>/dev/null || echo 0)
    if [ "$ok" = "1" ]; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "❌ Python 3.10~3.12가 필요합니다. 예: brew install python@3.11"
  exit 1
fi
echo "✅ 사용할 Python: $($PY --version) ($(command -v $PY))"

# 2) 가상환경 생성
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  echo "✅ .venv 생성"
fi
source .venv/bin/activate

# 3) 패키지 설치
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
if [ "$1" = "--stt" ]; then
  pip install -r requirements-stt.txt
fi

# 4) Jupyter 커널 등록
python -m ipykernel install --user --name middle_project_ai --display-name "middle_project_ai (.venv)"

# 5) .env 준비
[ -f .env ] || cp .env.example .env

# 6) 환경 점검
python scripts/check_env.py
echo ""
echo "🎉 완료. 이후에는 'source .venv/bin/activate' 로 가상환경을 켜세요."
