# data/custom — 팀 자체 구축 데이터

지금 들어 있는 파일은 **형식 확인용 샘플**입니다. 실제 학습 전 팀 데이터로 교체하세요.

| 파일 | 형식 | 목표 규모 | 쓰는 곳 |
|---|---|---|---|
| risk_scenarios.csv | text,label(none/low/mid/high) | 400~600 | F6 분류기 학습 |
| hard_negatives.csv | text,label,note | 100~150 | F6 오탐 테스트 |
| gate_b_responses.csv | user,response,expected(safe/unsafe),pattern | 위험 4패턴×30~50 + 정상 100~200 | F4 게이트 B 테스트 |

- 게이트 B 위험 응답 중 **'방법 정보' 패턴은 실제 내용을 쓰지 않고 자리표시로 대체**합니다.
- 라벨은 2인 이상 독립 작성 후 일치도 확인, 심리학과 대학원생 검수.
- AI Hub 등 외부 원본 데이터는 `data/raw/`에 두세요(깃에 올리지 않음).
