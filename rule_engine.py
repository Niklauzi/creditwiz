"""
ERDE Rule Engine
Hard disqualification rules applied on ENGINEERED features,
BEFORE model scoring. A disqualified application never reaches the model.
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
        "description": "Excessive delinquencies",
        "check": lambda d: d.get("pub_rec", 0) >= 2,
        "reason": "Applicant has 2 or more derogatory public records.",
    },
    {
        "id": "DQ-002",
        "description": "Debt-to-income ratio critically high",
        "check": lambda d: (d.get("dti") or 0) > 60,
        "reason": "Debt-to-income ratio exceeds 60% — income cannot support additional debt.",
    },
    {
        "id": "DQ-003",
        "description": "Excessive recent credit inquiries",
        "check": lambda d: d.get("inq_last_6mths", 0) >= 6,
        "reason": "6 or more credit inquiries in the last 6 months indicates credit distress.",
    },
    {
        "id": "DQ-004",
        "description": "Loan amount extreme relative to income",
        "check": lambda d: (d.get("loan_to_monthly_income") or 0) > 10,
        "reason": "Loan amount exceeds 10x monthly income — repayment is not feasible.",
    },
    {
        "id": "DQ-005",
        "description": "Worst credit grade",
        "check": lambda d: d.get("grade_numeric", 0) == 7,
        "reason": "Applicant holds a Grade G credit rating — below minimum acceptable threshold.",
    },
    {
        "id": "DQ-006",
        "description": "High risk purpose with poor grade",
        "check": lambda d: d.get("purpose_risk_score", 0) >= 0.7 and d.get("grade_numeric", 0) >= 5,
        "reason": "High-risk loan purpose combined with poor credit grade (E or below).",
    },
    {
        "id": "DQ-007",
        "description": "Unverified with high DTI",
        "check": lambda d: d.get("verification_strength", 0) == 0 and (d.get("dti") or 0) > 40,
        "reason": "Income not verified and DTI exceeds 40% — insufficient evidence of repayment capacity.",
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