import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional

app = FastAPI(title="ERDE")
templates = Jinja2Templates(directory="templates")

from logger import log_prediction
# --- Load artifacts ---

model = joblib.load("model.pkl")

preprocessor = joblib.load("preprocessor.pkl")

background_data = joblib.load("background.pkl") 

ACCEPT_THRESHOLD = 0.35
REVIEW_THRESHOLD = 0.60

CATEGORICAL_FIELDS = {"home_ownership", "grade"}


def get_decision(prob: float) -> tuple[str, str]:
    if prob < ACCEPT_THRESHOLD:
        return "ACCEPT", "accept"
    elif prob < REVIEW_THRESHOLD:
        return "REVIEW", "review"
    return "REJECT", "reject"


def run_inference(form_data: dict) -> dict:
    df = pd.DataFrame([form_data])

    # Cast types
    for col in df.columns:
        if col not in CATEGORICAL_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0][1])
    decision, css_class = get_decision(prob)

    # SHAP
    try:
        def model_predict(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(model_predict, background_data)
        shap_vals = explainer.shap_values(X, nsamples=100)
        sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

        feat_names = [n.split('__')[-1] for n in preprocessor.get_feature_names_out()]

        top_shap = sorted(
            [{"feature": k, "value": round(float(v), 4), "pos": float(v) > 0}
             for k, v in zip(feat_names, sv)],
            key=lambda x: abs(x["value"]),
            reverse=True
        )[:10]

        max_abs = max(abs(x["value"]) for x in top_shap) or 1
        for x in top_shap:
            x["pct"] = round(abs(x["value"]) / max_abs * 100)

    except Exception as e:
        top_shap = [{"feature": f"shap_error: {e}", "value": 0, "pos": False, "pct": 0}]

    return {
        "prob": round(prob * 100, 1),
        "prob_bar": round(prob * 100),
        "decision": decision,
        "css_class": css_class,
        "shap": top_shap,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    # Core loan
    loan_amnt: float = Form(...),
    term: int = Form(...),
    int_rate: float = Form(...),
    installment: float = Form(...),
    grade: str = Form(...),
    grade_numeric: int = Form(...),
    purpose_risk_score: float = Form(...),
    # Borrower
    annual_inc: float = Form(...),
    dti: Optional[float] = Form(None),
    debt_to_income_ratio: float = Form(...),
    loan_to_monthly_income: float = Form(...),
    home_ownership: str = Form(...),
    stable_employment: int = Form(...),
    high_dti_flag: int = Form(...),
    # Credit history
    open_acc: float = Form(...),
    total_acc: float = Form(...),
    inq_last_6mths: float = Form(...),
    inquiry_pressure: float = Form(...),
    revol_util: Optional[float] = Form(None),
    verification_strength: int = Form(...),
    strong_verification: int = Form(...),
    high_risk_grade: int = Form(...),
    # Behavioral
    airtime_frequency: float = Form(...),
    betting_frequency: float = Form(...),
    uses_savings: int = Form(...),
    savings_frequency: float = Form(...),
    savings_avg_amount: float = Form(...),
    shortterm_loans_avg_amount: float = Form(...),
    uses_utilities: int = Form(...),
    utilities_monthly_spend: float = Form(...),
):
    form_data = {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "grade": grade, "grade_numeric": grade_numeric,
        "purpose_risk_score": purpose_risk_score, "annual_inc": annual_inc,
        "dti": dti, "debt_to_income_ratio": debt_to_income_ratio,
        "loan_to_monthly_income": loan_to_monthly_income, "home_ownership": home_ownership,
        "stable_employment": stable_employment, "high_dti_flag": high_dti_flag,
        "open_acc": open_acc, "total_acc": total_acc, "inq_last_6mths": inq_last_6mths,
        "inquiry_pressure": inquiry_pressure, "revol_util": revol_util,
        "verification_strength": verification_strength, "strong_verification": strong_verification,
        "high_risk_grade": high_risk_grade, "airtime_frequency": airtime_frequency,
        "betting_frequency": betting_frequency, "uses_savings": uses_savings,
        "savings_frequency": savings_frequency, "savings_avg_amount": savings_avg_amount,
        "shortterm_loans_avg_amount": shortterm_loans_avg_amount,
        "uses_utilities": uses_utilities, "utilities_monthly_spend": utilities_monthly_spend,
    }

    try:
        result = run_inference(form_data)
        log_prediction(form_data, result)
        error = None
    except Exception as e:
        result = None
        error = str(e)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "error": error,
        "form_data": form_data,
    })
