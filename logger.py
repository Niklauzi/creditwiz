import os
import logging
from datetime import datetime

import sqlite3


LOG_DIR = "logs"
DB_PATH = "erde_predictions.db"



def get_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(LOG_DIR, f"prediction_{timestamp}.log")

    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    return logger


def log_prediction(form_data: dict, result: dict) -> None:
    logger = get_logger()

    logger.info("=== ERDE PREDICTION LOG ===")
    logger.info(f"Decision       : {result['decision']}")
    logger.info(f"Probability    : {result['prob']}%")
    logger.info("--- Input Features ---")
    logger.info(f"  predicted_loan_status: {1 if result['prob'] >= 50 else 0}")
    for k, v in form_data.items():
        logger.info(f"  {k}: {v}")
    logger.info("--- SHAP Attribution (Top Features) ---")
    for s in result["shap"]:
        direction = "↑ risk" if s["pos"] else "↓ risk"
        logger.info(f"  {s['feature']}: {s['value']:+.4f} ({direction})")
    logger.info("=== END ===")



def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            decision TEXT,
            probability REAL,
            predicted_loan_status TEXT,
            rule_id TEXT,
            disqualification_reason TEXT,
            loan_amnt REAL, term REAL, int_rate REAL, annual_inc REAL,
            dti REAL, inq_last_6mths REAL, open_acc REAL, revol_util REAL,
            total_acc REAL, installment REAL, debt_to_income_ratio REAL,
            loan_to_monthly_income REAL, high_dti_flag INTEGER,
            stable_employment INTEGER, inquiry_pressure REAL,
            grade_numeric INTEGER, high_risk_grade INTEGER,
            purpose_risk_score REAL, verification_strength INTEGER,
            strong_verification INTEGER, airtime_frequency REAL,
            betting_frequency REAL, uses_savings INTEGER,
            savings_frequency REAL, savings_avg_amount REAL,
            shortterm_loans_avg_amount REAL, uses_utilities INTEGER,
            utilities_monthly_spend REAL, home_ownership TEXT, grade TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(form_data: dict, result: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, decision, probability, predicted_loan_status,
            rule_id, disqualification_reason,
            loan_amnt, term, int_rate, annual_inc, dti, inq_last_6mths,
            open_acc, revol_util, total_acc, installment,
            debt_to_income_ratio, loan_to_monthly_income, high_dti_flag,
            stable_employment, inquiry_pressure, grade_numeric,
            high_risk_grade, purpose_risk_score, verification_strength,
            strong_verification, airtime_frequency, betting_frequency,
            uses_savings, savings_frequency, savings_avg_amount,
            shortterm_loans_avg_amount, uses_utilities,
            utilities_monthly_spend, home_ownership, grade
        ) VALUES (
            :timestamp, :decision, :probability, :predicted_loan_status,
            :rule_id, :disqualification_reason,
            :loan_amnt, :term, :int_rate, :annual_inc, :dti, :inq_last_6mths,
            :open_acc, :revol_util, :total_acc, :installment,
            :debt_to_income_ratio, :loan_to_monthly_income, :high_dti_flag,
            :stable_employment, :inquiry_pressure, :grade_numeric,
            :high_risk_grade, :purpose_risk_score, :verification_strength,
            :strong_verification, :airtime_frequency, :betting_frequency,
            :uses_savings, :savings_frequency, :savings_avg_amount,
            :shortterm_loans_avg_amount, :uses_utilities,
            :utilities_monthly_spend, :home_ownership, :grade
        )
    """, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision": result["decision"],
        "probability": result["prob"],
        "predicted_loan_status": "DEFAULT" if result["prob"] >= 50 else "FULLY PAID",
        "rule_id": result.get("rule_id"),
        "disqualification_reason": result.get("disqualification_reason"),
        **{k: form_data.get(k) for k in [
            "loan_amnt", "term", "int_rate", "annual_inc", "dti",
            "inq_last_6mths", "open_acc", "revol_util", "total_acc",
            "installment", "debt_to_income_ratio", "loan_to_monthly_income",
            "high_dti_flag", "stable_employment", "inquiry_pressure",
            "grade_numeric", "high_risk_grade", "purpose_risk_score",
            "verification_strength", "strong_verification", "airtime_frequency",
            "betting_frequency", "uses_savings", "savings_frequency",
            "savings_avg_amount", "shortterm_loans_avg_amount", "uses_utilities",
            "utilities_monthly_spend", "home_ownership", "grade"
        ]}
    })
    conn.commit()
    conn.close()