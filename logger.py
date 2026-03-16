import os
import sqlite3
import logging
from datetime import datetime

LOG_DIR = "logs"
DB_PATH = "erde_predictions.db"


def get_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(LOG_DIR, f"prediction_{timestamp}.log")
    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def log_prediction(engineered: dict, result: dict) -> None:
    logger = get_logger()
    logger.info("=== ERDE PREDICTION LOG ===")

    if result.get("disqualified"):
        logger.info("Decision       : HARD REJECT (Rule Engine)")
        logger.info(f"Rule ID        : {result['rule_id']}")
        logger.info(f"Reason         : {result['disqualification_reason']}")
    else:
        logger.info(f"Decision       : {result['decision']}")
        logger.info(f"Probability    : {result['prob']}%")
        logger.info(f"Loan Status    : {'DEFAULT' if result['prob'] >= 50 else 'FULLY PAID'}")
        logger.info("--- SHAP Attribution ---")
        for s in result["shap"]:
            direction = "INCREASES RISK" if s["pos"] else "REDUCES RISK"
            logger.info(f"  {s['feature']}: {s['value']:+.4f} ({direction})")

    logger.info("--- Engineered Features ---")
    for k, v in engineered.items():
        logger.info(f"  {k}: {v}")
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
            loan_amnt REAL, term INTEGER, int_rate REAL, installment REAL,
            annual_inc REAL, dti REAL, total_acc REAL, inq_last_6mths REAL,
            pub_rec REAL, revol_bal REAL, home_ownership TEXT,
            grade_numeric INTEGER, loan_to_monthly_income REAL,
            stable_employment INTEGER, long_term_loan INTEGER,
            verification_strength INTEGER, purpose_risk_score REAL
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(engineered: dict, result: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, decision, probability, predicted_loan_status,
            rule_id, disqualification_reason,
            loan_amnt, term, int_rate, installment, annual_inc, dti,
            total_acc, inq_last_6mths, pub_rec, revol_bal, home_ownership,
            grade_numeric, loan_to_monthly_income, stable_employment,
            long_term_loan, verification_strength, purpose_risk_score
        ) VALUES (
            :timestamp, :decision, :probability, :predicted_loan_status,
            :rule_id, :disqualification_reason,
            :loan_amnt, :term, :int_rate, :installment, :annual_inc, :dti,
            :total_acc, :inq_last_6mths, :pub_rec, :revol_bal, :home_ownership,
            :grade_numeric, :loan_to_monthly_income, :stable_employment,
            :long_term_loan, :verification_strength, :purpose_risk_score
        )
    """, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision": result["decision"],
        "probability": result.get("prob"),
        "predicted_loan_status": "DEFAULT" if (result.get("prob") or 100) >= 50 else "FULLY PAID",
        "rule_id": result.get("rule_id"),
        "disqualification_reason": result.get("disqualification_reason"),
        **{k: engineered.get(k) for k in [
            "loan_amnt", "term", "int_rate", "installment", "annual_inc",
            "dti", "total_acc", "inq_last_6mths", "pub_rec", "revol_bal",
            "home_ownership", "grade_numeric", "loan_to_monthly_income",
            "stable_employment", "long_term_loan", "verification_strength",
            "purpose_risk_score"
        ]}
    })
    conn.commit()
    conn.close()