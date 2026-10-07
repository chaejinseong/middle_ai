# middle_project_ai — 위험 감지·대응 AI 학습/테스트 환경

정신건강 비대면 추적관리 · 위기 판단 및 병원 인계 프로젝트의 **위험 감지·대응 기능**(F2·F4·F5·F6) 모델을
미리 학습하고 테스트하기 위한 환경입니다. 모델·데이터 목록은 프로젝트 문서 `위험감지대응_모델_데이터셋_표.md` 기준.

## 1. 설치 (처음 한 번)

```bash
cd ~/middle_project_ai
./setup.sh            # .venv 생성 + 패키지 설치 + Jupyter 커널 등록 + 환경 점검
./setup.sh --stt      # STT(Whisper)까지 설치하려면 (먼저 brew install ffmpeg)
```
- Python 3.10~3.12 필요 (없으면 `brew install python@3.11`)
- Apple Silicon 맥은 PyTorch **MPS** 가속을 자동 사용, NVIDIA는 CUDA, 그 외 CPU
- `.env`에 API 키(HF_TOKEN, UPSTAGE_API_KEY 등) 입력

모델 받기 (설치 후 한 번):
```bash
python scripts/download_models.py           # 필수 모델 약 2GB
python scripts/download_models.py --large   # 대형 가드레일까지 (+약 25GB)
python scripts/download_models.py --list    # 받은 모델 확인
```

이후 작업할 때마다:
```bash
cd ~/middle_project_ai && source .venv/bin/activate
```
VS Code/Jupyter에서는 커널 **"middle_project_ai (.venv)"** 선택.

## 1-1. Colab에서 실행 (권장 — 맥 저장공간 절약)

구글 드라이브 `middle_project_ai_colab/` 폴더의 `colab_risk_models.ipynb`를 열고(연결 앱: Google Colaboratory)
런타임을 T4 GPU로 바꾼 뒤 위에서부터 실행. 코드는 드라이브의 `middle_project_ai_code.zip`에서 자동으로 풀림.
모델은 Colab 임시 디스크에 받으므로 맥·드라이브 용량을 쓰지 않음.

## 2. 폴더 구조

```
middle_project_ai/
├── setup.sh                  가상환경 설치 스크립트
├── requirements.txt          공통 패키지 / requirements-stt.txt (선택)
├── configs/
│   ├── risk_classifier.yaml  F6 분류기 학습 설정
│   └── risk_lexicon.yaml     F6 규칙 사전 (Tier1~3, 부정·과장 처리)
├── data/
│   ├── raw/                  AI Hub 등 외부 원본 (git 제외)
│   ├── processed/            전처리 결과 (git 제외)
│   └── custom/               팀 자체 데이터 (지금은 형식 샘플)
├── src/
│   ├── common/               장치 선택(MPS/CUDA/CPU), 경로·라벨
│   ├── risk/                 F6 규칙 스코어러 · 분류기 학습 · 평가
│   ├── guard/                F4 게이트 A/B 가드레일 모델 테스트
│   ├── trend/                F5 trend_flag + 합성 환자 궤적
│   └── pii/                  F2 개인정보 마스킹
├── scripts/check_env.py      환경 점검
├── scripts/download_models.py  필요한 모델 일괄 다운로드
├── models/                   학습된 모델 저장 (git 제외)
├── outputs/                  평가 결과 CSV/JSON (git 제외)
└── notebooks/
```

## 3. 실행 방법

| 기능 | 명령 | 비고 |
|---|---|---|
| 환경 점검 | `python scripts/check_env.py` | |
| F6 규칙 스코어 | `python -m src.risk.rules "요즘 다 끝내고 싶어"` | 사전은 `configs/risk_lexicon.yaml` |
| F6 분류기 학습 | `python -m src.risk.train_classifier` | KcELECTRA 첫 실행 시 약 500MB 다운로드 |
| F6 평가 (규칙/분류기/결합 비교) | `python -m src.risk.evaluate --mode all` | `--test data/custom/risk_scenarios.csv` 로 바꿔 가능 |
| F4 가드레일 테스트 | `python -m src.guard.run_guard --model qwen0.6b` | `--model kanana` (8B, 메모리 16GB+ 필요) / `qwen4b` |
| F5 추세 탐지 | `python -m src.trend.trend_flag --simulate 100` | 합성 환자로 탐지율·지연·오경보 |
| F2 PII 마스킹 | `python -m src.pii.mask "김민수 010-1234-5678"` | `--no-model` 은 정규식만 |

결과는 모두 `outputs/` 아래 날짜별 폴더/CSV로 저장됩니다.

## 4. 데이터 넣기

`data/custom/README.md` 참고. 지금 CSV들은 **형식 확인용 샘플(수십 줄)**이라 분류기 성능은 의미가 없습니다.
팀 라벨 데이터(400~600문장)로 교체한 뒤 학습하세요.

## 5. 메모

- **Kanana Safeguard 8B**: bf16 약 16GB. 맥은 통합 메모리 24GB 이상 권장, 부족하면 Colab(GPU)에서 실행하거나 `qwen0.6b`로 먼저 테스트.
- 규칙 기준선(샘플): "과제 때문에 죽고 싶다ㅋㅋ"가 중위험으로 잡힘 → 분류기·결합으로 개선해야 할 대표 오탐 사례.
- trend_flag 기본값(합성 100명 기준): 오경보 0%, 점진 악화 평균 13일·급변 2~3일 만에 탐지. 악화 후 미접속 환자는 탐지율이 낮아 F11 미접속 신호와 함께 써야 함.
