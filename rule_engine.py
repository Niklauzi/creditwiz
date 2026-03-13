"""
ERDE Rule Engine
Hard disqualification rules applied BEFORE preprocessing and model scoring.
A disqualified application never reaches the model.
Auto-qualification flags are advisory only — model still scores.
"""

from dataclasses import dataclass


@dataclass
class RuleResult:
    passed: bool          # True = proceed to model, False = hard reject
    reason: str | None    # Disqualification reason if failed
    rule_id: str | None   # Rule identifier for audit log


DISQUALIFICATION_RULES = [
    {
        "id": "DQ-001",
        "description": "Excessive delinquencies in last 2 years",
        "check": lambda d: d.get("delinq_2yrs", 0) >= 3,
        "reason": "Applicant has 3 or more delinquencies in the past 2 years.",
    },
    {
        "id": "DQ-002",
        "description": "Multiple public derogatory records",
        "check": lambda d: d.get("pub_rec", 0) >= 2,
        "reason": "Applicant has 2 or more derogatory public records.",
    },
    {
        "id": "DQ-003",
        "description": "Debt-to-income ratio critically high",
        "check": lambda d: (d.get("dti") or 0) > 60,
        "reason": "Debt-to-income ratio exceeds 60% — income cannot support additional debt.",
    },
    {
        "id": "DQ-004",
        "description": "Excessive recent credit inquiries",
        "check": lambda d: d.get("inq_last_6mths", 0) >= 6,
        "reason": "6 or more credit inquiries in the last 6 months indicates credit distress.",
    },
    {
        "id": "DQ-005",
        "description": "Revolving utilization critically high",
        "check": lambda d: (d.get("revol_util") or 0) >= 95,
        "reason": "Revolving credit utilization at or above 95% — borrower is fully maxed out.",
    },
    {
        "id": "DQ-006",
        "description": "Loan amount extreme relative to income",
        "check": lambda d: (d.get("loan_to_monthly_income") or 0) > 10,
        "reason": "Loan amount exceeds 10x monthly income — repayment is not feasible.",
    },
    {
        "id": "DQ-007",
        "description": "Pathological betting behaviour",
        "check": lambda d: (d.get("betting_frequency") or 0) > 20,
        "reason": "Betting frequency exceeds acceptable threshold — high financial risk behaviour.",
    },
    {
        "id": "DQ-008",
        "description": "Worst credit grade",
        "check": lambda d: d.get("grade", "").upper() == "G",
        "reason": "Applicant holds a Grade G credit rating — below minimum acceptable threshold.",
    },
]


def run_rules(form_data: dict) -> RuleResult:
    """
    Run all disqualification rules against form data.
    Returns on first failure — no need to check the rest.
    """
    for rule in DISQUALIFICATION_RULES:
        try:
            if rule["check"](form_data):
                return RuleResult(
                    passed=False,
                    reason=rule["reason"],
                    rule_id=rule["id"],
                )
        except Exception:
            continue  # malformed input for this rule — skip, let model handle it

    return RuleResult(passed=True, reason=None, rule_id=None)
