"""F6 위험도 판단 — 1차 규칙기반 스코어러 (Tier 1~3 사전 + 부정·과거·과장 처리)

사전은 configs/risk_lexicon.yaml 에서 읽음. 팀이 사전을 고치면 코드 수정 없이 반영됨.
점수 = Σ(티어 가중치 × 강도계수) × 완화계수, 0~100으로 자름.
"""
import re
from dataclasses import dataclass, field
from src.common.io import load_config


@dataclass
class RuleResult:
    risk_score: float
    risk_level: str
    trigger_evidence: list = field(default_factory=list)


class RuleScorer:
    def __init__(self, lexicon_path="configs/risk_lexicon.yaml"):
        cfg = load_config(lexicon_path)
        self.tiers = cfg["tiers"]
        self.negation = [re.compile(p) for p in cfg["negation_after"]]
        self.past = [re.compile(p) for p in cfg["past_markers"]]
        self.recovered = [re.compile(p) for p in cfg["recovered_markers"]]
        self.hyperbole = [re.compile(p) for p in cfg["hyperbole"]]
        self.intensifiers = cfg["intensifiers"]
        self.laughter = re.compile(cfg["laughter"])
        self.th = cfg["thresholds"]
        for t in self.tiers.values():
            t["_re"] = [re.compile(p) for p in t["patterns"]]

    def _negated(self, text, end):
        window = text[end:end + 12]
        return any(r.match(window) for r in self.negation)

    def score(self, text: str) -> RuleResult:
        evidence, total = [], 0.0
        for tier_name, t in self.tiers.items():
            for rx in t["_re"]:
                for m in rx.finditer(text):
                    if self._negated(text, m.end()):
                        evidence.append({"tier": tier_name, "match": m.group(), "ignored": "negation"})
                        continue
                    w = t["weight"]
                    if any(i in text for i in self.intensifiers):
                        w *= 1.2
                    total += w
                    evidence.append({"tier": tier_name, "match": m.group(), "weight": round(w, 1)})

        # 완화: 과장 관용표현 / 웃음 / 과거형+현재 회복
        if any(h.search(text) for h in self.hyperbole) and not any(
            e.get("tier") == "tier1" and "ignored" not in e for e in evidence
        ):
            total *= 0.2
        if self.laughter.search(text):
            total *= 0.5
        if any(p.search(text) for p in self.past) and any(r.search(text) for r in self.recovered):
            total *= 0.3

        score = min(100.0, round(total, 1))
        if score >= self.th["high"]:
            level = "high"
        elif score >= self.th["mid"]:
            level = "mid"
        elif score >= self.th["low"]:
            level = "low"
        else:
            level = "none"
        return RuleResult(score, level, evidence)


if __name__ == "__main__":
    import sys
    s = RuleScorer()
    for line in (sys.argv[1:] or ["요즘 그냥 다 끝내고 싶어", "배고파 죽겠다", "죽고 싶지 않아"]):
        print(line, "→", s.score(line))
