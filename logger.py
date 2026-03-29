import os
import logging
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = "logs"
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = DATABASE_URL is not None

# Model feature columns stored in DB
DB_COLS = [
    "loan_amnt", "term", "int_rate", "dti", "inq_last_6mths", "delinq_2yrs",
    "home_ownership", "loan_to_monthly_income", "very_high_utilization",
    "long_term_loan", "verification_strength", "purpose_risk_score"
]


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
    return conn


def ph():
    return "%s" if USE_POSTGRES else "?"


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    id_col = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS predictions (
            id {id_col},
            timestamp TEXT,
            decision TEXT,
            probability REAL,
            predicted_loan_status TEXT,
            rule_id TEXT,
            disqualification_reason TEXT,
            loan_amnt REAL,
            term INTEGER,
            int_rate REAL,
            dti REAL,
            inq_last_6mths REAL,
            delinq_2yrs REAL,
            home_ownership TEXT,
            loan_to_monthly_income REAL,
            very_high_utilization INTEGER,
            long_term_loan INTEGER,
            verification_strength INTEGER,
            purpose_risk_score REAL
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
            loan_amnt, term, int_rate, dti, inq_last_6mths, delinq_2yrs,
            home_ownership, loan_to_monthly_income, very_high_utilization,
            long_term_loan, verification_strength, purpose_risk_score
        ) VALUES (
            {p},{p},{p},{p},{p},{p},
            {p},{p},{p},{p},{p},{p},
            {p},{p},{p},{p},{p},{p}
        )
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        result["decision"],
        result.get("prob"),
        "DEFAULT" if (result.get("prob") or 100) >= 50 else "FULLY PAID",
        result.get("rule_id"),
        result.get("disqualification_reason"),
        engineered.get("loan_amnt"), engineered.get("term"),
        engineered.get("int_rate"), engineered.get("dti"),
        engineered.get("inq_last_6mths"), engineered.get("delinq_2yrs"),
        engineered.get("home_ownership"), engineered.get("loan_to_monthly_income"),
        engineered.get("very_high_utilization"), engineered.get("long_term_loan"),
        engineered.get("verification_strength"), engineered.get("purpose_risk_score"),
    ))
    conn.commit()
    cur.close()
    conn.close()


def fetch_all_predictions() -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = [dict(zip([col[0] for col in cur.description], row)) for row in cur.fetchall()]
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