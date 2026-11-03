"""Rules engine: combines the model score with configurable business rules.

Model score -> risk level -> default decision:
    low    (p <  review threshold)           -> approve
    medium (review <= p < block threshold)   -> review (goes to the analyst case queue)
    high   (p >= block threshold)            -> block

Business rules can then escalate a decision (never silently downgrade it).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

DECISION_FOR_LEVEL = {"low": "approve", "medium": "review", "high": "block"}
_ORDER = {"approve": 0, "review": 1, "block": 2}
_LEVEL_FOR_DECISION = {"approve": "low", "review": "medium", "block": "high"}


@dataclass
class RuleConfig:
    hard_block_amount: float = 500_000.0   # INR; block when a new device spends this much or more
    velocity_review_count: int = 12        # review when this many transactions occurred in 24h
    blocklist: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Optional[str]) -> "RuleConfig":
        if path and Path(path).exists():
            return cls(**json.loads(Path(path).read_text()))
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


def decide(
    probability: float,
    t_review: float,
    t_block: float,
    feats: dict,
    customer_id: str,
    amount: float,
    rules: RuleConfig,
) -> dict:
    level = "high" if probability >= t_block else "medium" if probability >= t_review else "low"
    decision = DECISION_FOR_LEVEL[level]
    reasons = [f"model fraud probability {probability:.3f} -> {level} risk"]

    def escalate(to: str, why: str) -> None:
        nonlocal decision
        if _ORDER[to] > _ORDER[decision]:
            decision = to
        reasons.append(why)

    if customer_id in set(rules.blocklist):
        escalate("block", "rule: customer is on the blocklist")
    if feats.get("new_device") == 1 and amount >= rules.hard_block_amount:
        escalate("block", f"rule: new device with amount >= {rules.hard_block_amount:,.0f}")
    if feats.get("txn_count_24h", 0) >= rules.velocity_review_count:
        escalate("review", f"rule: {feats['txn_count_24h']} transactions in the last 24h")

    return {"decision": decision, "risk_level": _LEVEL_FOR_DECISION[decision], "reasons": reasons}
