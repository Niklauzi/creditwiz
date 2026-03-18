import os
import logging
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from psycopg2.extras import RealDictCursor

load_dotenv()

LOG_DIR = "logs"
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = DATABASE_URL is not None


def get_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(LOG_DIR, f"prediction_{timestamp}.log")
    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def get_conn():
    if USE_POSTGRES:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    conn = sqlite3.connect("erde_predictions.db")
    conn.row_factory = sqlite3.Row
    return conn


def get_dict_cursor(conn):
    """Get a cursor that returns dictionaries"""
    if USE_POSTGRES:
        return conn.cursor(cursor_factory=RealDictCursor)
    return conn.cursor()


def get_dict_cursor(conn):
    """Get a cursor that returns dictionaries"""
    if USE_POSTGRES:
        return conn.cursor(cursor_factory=RealDictCursor)
    return conn.cursor()


def ph():
    return "%s" if USE_POSTGRES else "?"


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    if USE_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                decision TEXT, probability REAL, predicted_loan_status TEXT,
                rule_id TEXT, disqualification_reason TEXT,
                loan_amnt REAL, term INTEGER, int_rate REAL, installment REAL,
                annual_inc REAL, dti REAL, total_acc REAL, inq_last_6mths REAL,
                pub_rec REAL, revol_bal REAL, home_ownership TEXT,
                grade_numeric INTEGER, loan_to_monthly_income REAL,
                stable_employment INTEGER, long_term_loan INTEGER,
                verification_strength INTEGER, purpose_risk_score REAL
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                decision TEXT, probability REAL, predicted_loan_status TEXT,
                rule_id TEXT, disqualification_reason TEXT,
                loan_amnt REAL, term INTEGER, int_rate REAL, installment REAL,
                annual_inc REAL, dti REAL, total_acc REAL, inq_last_6mths REAL,
                pub_rec REAL, revol_bal REAL, home_ownership TEXT,
                grade_numeric INTEGER, loan_to_monthly_income REAL,
                stable_employment INTEGER, long_term_loan INTEGER,
                verification_strength INTEGER, purpose_risk_score REAL
            )
        """)
    conn.commit()
    cur.close()
    conn.close()


def save_to_db(engineered: dict, result: dict):
    conn = get_conn()
    cur = conn.cursor()
    p = ph()
    cur.execute(f"""
        INSERT INTO predictions (
            timestamp, decision, probability, predicted_loan_status,
            rule_id, disqualification_reason,
            loan_amnt, term, int_rate, installment, annual_inc, dti,
            total_acc, inq_last_6mths, pub_rec, revol_bal, home_ownership,
            grade_numeric, loan_to_monthly_income, stable_employment,
            long_term_loan, verification_strength, purpose_risk_score
        ) VALUES (
            {p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},
            {p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p}
        )
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result["decision"],
        result.get("prob"),
        "DEFAULT" if (result.get("prob") or 100) >= 50 else "FULLY PAID",
        result.get("rule_id"),
        result.get("disqualification_reason"),
        engineered.get("loan_amnt"), engineered.get("term"),
        engineered.get("int_rate"), engineered.get("installment"),
        engineered.get("annual_inc"), engineered.get("dti"),
        engineered.get("total_acc"), engineered.get("inq_last_6mths"),
        engineered.get("pub_rec"), engineered.get("revol_bal"),
        engineered.get("home_ownership"), engineered.get("grade_numeric"),
        engineered.get("loan_to_monthly_income"), engineered.get("stable_employment"),
        engineered.get("long_term_loan"), engineered.get("verification_strength"),
        engineered.get("purpose_risk_score"),
    ))
    conn.commit()
    cur.close()
    conn.close()


def fetch_all_predictions() -> list[dict]:
    conn = get_conn()
    cur = get_dict_cursor(conn)
    cur.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


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