from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

LABELS = ["none", "low", "mid", "high"]          # 위험도 4클래스
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_config(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
