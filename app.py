import io
import csv
import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import Optional

from logger import log_prediction, init_db, save_to_db, fetch_all_predictions
from rule_engine import run_rules

app = FastAPI(title="ERDE")
templates = Jinja2Templates(directory="templates")

init_db()

# --- Load artifacts ---
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
background_data = joblib.load("background.pkl")

ACCEPT_THRESHOLD = 0.35
REVIEW_THRESHOLD = 0.60

GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

PURPOSE_RISK_MAP = {
    'credit_card': 0.2, 'car': 0.2, 'home_improvement': 0.2,
    'major_purchase': 0.3, 'medical': 0.3, 'wedding': 0.3,
    'vacation': 0.4, 'moving': 0.4, 'house': 0.4,
    'debt_consolidation': 0.5, 'other': 0.5,
    'small_business': 0.7, 'renewable_energy': 0.5,
    'educational': 0.4
}

VERIFICATION_MAP = {
    'Not Verified': 0, 'Source Verified': 1, 'Verified': 2
}


def engineer_features(raw: dict) -> dict:
    loan_amnt = float(raw['loan_amnt'])
    annual_inc = float(raw['annual_inc'])
    term = int(raw['term'])
    grade = raw['grade'].upper()
    emp_length = float(raw['emp_length'])
    purpose = raw['purpose'].lower()
    verification_status = raw['verification_status']
    monthly_income = annual_inc / 12 if annual_inc > 0 else 1

    return {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': float(raw['int_rate']),
        'installment': float(raw['installment']),
        'annual_inc': annual_inc,
        'dti': float(raw['dti']) if raw.get('dti') else 0.0,
        'total_acc': float(raw['total_acc']),
        'inq_last_6mths': float(raw['inq_last_6mths']),
        'pub_rec': float(raw['pub_rec']),
        'revol_bal': float(raw['revol_bal']),
        'home_ownership': raw['home_ownership'],
        'grade_numeric': GRADE_MAP.get(grade, 4),
        'loan_to_monthly_income': loan_amnt / monthly_income,
        'stable_employment': 1 if emp_length >= 2 else 0,
        'long_term_loan': 1 if term == 60 else 0,
        'verification_strength': VERIFICATION_MAP.get(verification_status, 0),
        'purpose_risk_score': PURPOSE_RISK_MAP.get(purpose, 0.5),
    }


def get_decision(prob: float) -> tuple[str, str]:
    if prob < ACCEPT_THRESHOLD:
        return "ACCEPT", "accept"
    elif prob < REVIEW_THRESHOLD:
        return "REVIEW", "review"
    return "REJECT", "reject"


def run_inference(engineered: dict) -> dict:
    df = pd.DataFrame([engineered])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    X = preprocessor.transform(df)
    prob = float(model.predict_proba(X)[0][1])
    decision, css_class = get_decision(prob)

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
        )

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


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    loan_amnt: float = Form(...),
    term: int = Form(...),
    int_rate: float = Form(...),
    installment: float = Form(...),
    purpose: str = Form(...),
    grade: str = Form(...),
    annual_inc: float = Form(...),
    emp_length: float = Form(...),
    home_ownership: str = Form(...),
    verification_status: str = Form(...),
    dti: Optional[float] = Form(None),
    total_acc: float = Form(...),
    inq_last_6mths: float = Form(...),
    pub_rec: float = Form(...),
    revol_bal: float = Form(...),
):
    raw_data = {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "purpose": purpose, "grade": grade,
        "annual_inc": annual_inc, "emp_length": emp_length,
        "home_ownership": home_ownership, "verification_status": verification_status,
        "dti": dti, "total_acc": total_acc, "inq_last_6mths": inq_last_6mths,
        "pub_rec": pub_rec, "revol_bal": revol_bal,
    }

    engineered = engineer_features(raw_data)
    rule_result = run_rules(engineered)

    if not rule_result.passed:
        result = {
            "prob": 100,
            "prob_bar": 100,
            "decision": "REJECT",
            "css_class": "reject",
            "shap": [],
            "disqualified": True,
            "disqualification_reason": rule_result.reason,
            "rule_id": rule_result.rule_id,
        }
        log_prediction(engineered, result)
        save_to_db(engineered, result)
        error = None
    else:
        try:
            result = run_inference(engineered)
            result["disqualified"] = False
            log_prediction(engineered, result)
            save_to_db(engineered, result)
            error = None
        except Exception as e:
            result = None
            error = str(e)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "error": error,
        "form_data": raw_data,
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    rows = fetch_all_predictions()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "rows": rows,
        "total": len(rows),
        "accepts": sum(1 for r in rows if r["decision"] == "ACCEPT"),
        "reviews": sum(1 for r in rows if r["decision"] == "REVIEW"),
        "rejects": sum(1 for r in rows if r["decision"] == "REJECT"),
    })


@app.get("/dashboard/export")
async def export_csv():
    rows = fetch_all_predictions()
    if not rows:
        return StreamingResponse(iter([""]), media_type="text/csv")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=erde_predictions.csv"}
    )
