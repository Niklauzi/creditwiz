import os
import logging
from datetime import datetime

LOG_DIR = "logs"


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
    for k, v in form_data.items():
        logger.info(f"  {k}: {v}")
    logger.info("--- SHAP Attribution (Top Features) ---")
    for s in result["shap"]:
        direction = "↑ risk" if s["pos"] else "↓ risk"
        logger.info(f"  {s['feature']}: {s['value']:+.4f} ({direction})")
    logger.info("=== END ===")