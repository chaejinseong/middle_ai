"""F2 개인정보 마스킹 — NER 모델(pii-ner-kcelectra) + 정규식

사용:  python -m src.pii.mask "저 김민수인데 010-1234-5678로 연락주세요"
       python -m src.pii.mask --no-model "..."   (정규식만)
"""
import argparse, re

REGEX = {
    "PHONE": re.compile(r"01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}"),
    "RRN": re.compile(r"\d{6}[-\s]?[1-4]\d{6}"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
}
MODEL = "kiyuyeon/pii-ner-kcelectra"


def mask_regex(text):
    for tag, rx in REGEX.items():
        text = rx.sub(f"[{tag}]", text)
    return text


class PIIMasker:
    def __init__(self, use_model=True):
        self.ner = None
        if use_model:
            from transformers import pipeline
            from src.common.device import get_device
            dev = get_device()
            self.ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple",
                                device=0 if dev == "cuda" else (dev if dev == "mps" else -1))

    def mask(self, text):
        if self.ner:
            ents = sorted(self.ner(text), key=lambda e: e["start"], reverse=True)
            for e in ents:
                text = text[:e["start"]] + f"[{e['entity_group']}]" + text[e["end"]:]
        return mask_regex(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    m = PIIMasker(use_model=not a.no_model)
    for t in a.text or ["저 김민수인데요 010-1234-5678로 연락 주세요. 서울시 마포구 살아요."]:
        print(t, "\n →", m.mask(t))
