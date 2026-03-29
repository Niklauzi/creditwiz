"""
ERDE Rule Engine
Hard disqualification on engineered features before model scoring.
"""

from dataclasses import dataclass


@dataclass
class RuleResult:
    passed: bool
    reason: str | None
    rule_id: str | None


DISQUALIFICATION_RULES = [
    {
        "id": "DQ-001",
        "description": "Debt-to-income ratio critically high",
        "check": lambda d: (d.get("dti") or 0) > 60,
        "reason": "Debt-to-income ratio exceeds 60% — income cannot support additional debt.",
    },
    {
        "id": "DQ-002",
        "description": "Excessive recent credit inquiries",
        "check": lambda d: d.get("inq_last_6mths", 0) >= 6,
        "reason": "6 or more credit inquiries in the last 6 months indicates credit distress.",
    },
    {
        "id": "DQ-003",
        "description": "Loan amount extreme relative to income",
        "check": lambda d: (d.get("loan_to_monthly_income") or 0) > 10,
        "reason": "Loan amount exceeds 10x monthly income — repayment is not feasible.",
    },
    {
        "id": "DQ-004",
        "description": "Repeated delinquencies",
        "check": lambda d: d.get("delinq_2yrs", 0) >= 3,
        "reason": "3 or more delinquencies in the past 2 years indicates persistent payment failure.",
    },
    {
        "id": "DQ-005",
        "description": "High risk purpose with unverified income",
        "check": lambda d: d.get("purpose_risk_score", 0) >= 0.7 and d.get("verification_strength", 0) == 0,
        "reason": "High-risk loan purpose with unverified income — insufficient evidence of repayment capacity.",
    },
    {
        "id": "DQ-006",
        "description": "Unverified income with high DTI",
        "check": lambda d: d.get("verification_strength", 0) == 0 and (d.get("dti") or 0) > 40,
        "reason": "Income not verified and DTI exceeds 40% — insufficient evidence of repayment capacity.",
    },
    {
        "id": "DQ-007",
        "description": "Revolving credit fully maxed out",
        "check": lambda d: d.get("very_high_utilization", 0) == 1 and (d.get("dti") or 0) > 40,
        "reason": "Revolving credit above 90% utilization combined with high DTI — severe over-leverage.",
    },
]


def run_rules(engineered: dict) -> RuleResult:
    for rule in DISQUALIFICATION_RULES:
        try:
            if rule["check"](engineered):
                return RuleResult(
                    passed=False,
                    reason=rule["reason"],
                    rule_id=rule["id"],
                )
        except Exception:
            continue
    return RuleResult(passed=True, reason=None, rule_id=None)