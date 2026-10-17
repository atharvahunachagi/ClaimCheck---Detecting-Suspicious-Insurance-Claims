from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# -------------------------------------------------------------------
# ClaimCheck | Insurance claim fraud screening dashboard
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "fraud_oracle.csv"
MODEL_PATH = BASE_DIR / "fraud_mlp.pt"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
TARGET = "FraudFound_P"

st.set_page_config(
    page_title="ClaimCheck | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: rgba(120, 130, 150, 0.08);
        border: 1px solid rgba(120, 130, 150, 0.20);
        padding: 16px 18px;
        border-radius: 14px;
    }
    .hero {
        padding: 1.2rem 1.4rem; border-radius: 16px;
        background: linear-gradient(115deg, rgba(28,65,110,.22), rgba(20,145,130,.10));
        border: 1px solid rgba(120,130,150,.25); margin-bottom: 1rem;
    }
    .muted {color: #8b93a1; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


class FraudMLP(nn.Module):
    """Architecture must match the one used to train fraud_mlp.pt."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


@st.cache_data
def load_dataset(path: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same engineered features documented in the training notebook."""
    result = frame.copy()
    age_map = {
        "new": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "more than 7": 8,
    }
    price_map = {
        "less than 20000": 1, "20000 to 29000": 2,
        "30000 to 39000": 3, "40000 to 59000": 4,
        "60000 to 69000": 5, "more than 69000": 6,
    }
    if "AgeOfVehicle" in result.columns and "VehiclePrice" in result.columns:
        result["VehiclePrice_to_Age"] = result["AgeOfVehicle"].map(age_map)
        result["Price_Ratio"] = (
            result["VehiclePrice"].map(price_map)
            / result["VehiclePrice_to_Age"].replace(0, np.nan)
        )
    return result


def model_assets_available() -> bool:
    return MODEL_PATH.is_file() and PREPROCESSOR_PATH.is_file()


@st.cache_resource
def load_preprocessor(path: str, modified_time: float):
    return joblib.load(path)


@st.cache_resource
def load_model_bundle(model_path, model_mtime, prep_path, prep_mtime):
    prep = joblib.load(prep_path)
    model = FraudMLP(input_dim=149)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, prep


def feature_frame(frame):
    return prepare_features(frame.drop(columns=[TARGET, "PolicyNumber"], errors="ignore"))


@st.cache_data(show_spinner="Scoring claims with the trained model…")
def score_dataset(data, model_mtime, prep_mtime):
    model, prep = load_model_bundle(str(MODEL_PATH), model_mtime, str(PREPROCESSOR_PATH), prep_mtime)
    x = np.asarray(prep.transform(feature_frame(data)), dtype=np.float32)
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(x))).squeeze(1).numpy()
    result = pd.DataFrame(index=data.index)
    result["Model score"] = scores
    result["Prediction"] = np.where(scores >= 0.5, "Potential fraud", "Likely legitimate")
    result["Risk"] = np.select([scores >= 0.70, scores >= 0.35], ["High risk", "Medium risk"], default="Low risk")
    return result


def predict_one(frame):
    model, prep = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime, str(PREPROCESSOR_PATH), PREPROCESSOR_PATH.stat().st_mtime)
    x = np.asarray(prep.transform(feature_frame(frame)), dtype=np.float32)
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.tensor(x))).item())


def get_risk_band(probability: float) -> str:
    # Initial operating bands for the interface; tune against validation data
    # before using these for real-world claim decisions.
    if probability >= 0.70:
        return "High risk"
    if probability >= 0.35:
        return "Medium risk"
    return "Low risk"


def show_header():
    st.markdown(
        '<div class="hero"><h1 style="margin:0">🛡️ ClaimCheck</h1>'
        '<p style="margin:.4rem 0 0">Insurance claim fraud intelligence · '
        'Model-assisted screening and investigation</p></div>',
        unsafe_allow_html=True,
    )


if not DATA_PATH.is_file():
    st.error("Dataset not found. Place fraud_oracle.csv beside app.py in the repository.")
    st.stop()

try:
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load fraud_oracle.csv: {exc}")
    st.stop()

if TARGET not in df.columns:
    st.error(f"Expected target column '{TARGET}' was not found in fraud_oracle.csv.")
    st.stop()

show_header()
assets_ready = model_assets_available()

with st.sidebar:
    st.title("ClaimCheck")
    st.caption("Fraud detection workspace")
    page = st.radio(
        "Navigate",
        ["Overview & Analytics", "Predict a Claim", "Fraud Risk Analysis", "Claim Investigation", "Model Performance"],
        label_visibility="visible",
    )
    st.divider()
    st.caption("DATASET")
    st.write(f"{len(df):,} historical records")
    st.caption("MODEL STATUS")
    if assets_ready:
        st.success("Model artifacts found")
    else:
        st.warning("Prediction model not connected")
        st.caption("Add fraud_mlp.pt and preprocessor.pkl to enable live predictions.")

# Historical target labels are not the same thing as current model predictions.
historical = df[TARGET].fillna(0).astype(int)
fraud_count = int((historical == 1).sum())
legit_count = int((historical == 0).sum())
fraud_pct = (fraud_count / len(df) * 100) if len(df) else 0

predictions = None
prediction_error = None
if assets_ready:
    try:
        predictions = score_dataset(df, MODEL_PATH.stat().st_mtime, PREPROCESSOR_PATH.stat().st_mtime)
    except Exception as exc:
        prediction_error = str(exc)

if page == "Overview & Analytics":
    st.subheader("Overview & analytics")
    st.caption("Historical claim data and model-generated screening results.")
    if prediction_error:
        st.error("The files exist, but the model could not score the dataset. The saved weights and preprocessor must come from the same training run.")
        with st.expander("Technical details"): st.code(prediction_error)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Claims analyzed", f"{len(df):,}")
    if predictions is not None:
        high = int((predictions["Risk"] == "High risk").sum())
        low = int((predictions["Risk"] == "Low risk").sum())
        med = int((predictions["Risk"] == "Medium risk").sum())
        k2.metric("High-risk claims", f"{high:,}")
        k3.metric("Low-risk claims", f"{low:,}")
        k4.metric("Model status", "Operational")
        st.caption(f"{med:,} claims fall in the medium-risk band. Risk bands use uncalibrated model scores, not verified fraud probabilities.")
    else:
        k2.metric("High-risk claims", "—")
        k3.metric("Low-risk claims", "—")
        k4.metric("Model status", "Not connected" if not assets_ready else "Inference error")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Historical labels")
        st.bar_chart(pd.DataFrame({"Claims": [legit_count, fraud_count]}, index=["Legitimate", "Fraud-labelled"]), horizontal=True)
        st.caption(f"{fraud_pct:.2f}% of source records have a fraud label. This is ground-truth data, not a model prediction.")
    with right:
        st.markdown("#### Model risk distribution")
        if predictions is not None:
            st.bar_chart(predictions["Risk"].value_counts().reindex(["High risk", "Medium risk", "Low risk"], fill_value=0), horizontal=True)
        else:
            st.info("Risk distribution appears after model inference succeeds.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Claims by vehicle make")
        if "Make" in df.columns: st.bar_chart(df["Make"].fillna("Unknown").value_counts().head(10), horizontal=True)
    with right:
        st.markdown("#### Historical labels by accident area")
        if "AccidentArea" in df.columns:
            st.bar_chart(pd.crosstab(df["AccidentArea"].fillna("Unknown"), historical).rename(columns={0:"Legitimate",1:"Fraud-labelled"}))
    st.markdown("#### High-risk review queue")
    if predictions is not None:
        queue = df.loc[predictions.index[predictions["Risk"] == "High risk"]].copy()
        queue.insert(0, "Claim ID", [f"CL-{i+1:05d}" for i in queue.index])
        queue["Risk"] = predictions.loc[queue.index, "Risk"]
        queue["Model score"] = (predictions.loc[queue.index, "Model score"] * 100).round(1)
        cols = ["Claim ID", "Risk", "Model score"] + [c for c in ["Make", "AccidentArea", "Fault", "VehiclePrice", "AgeOfVehicle"] if c in queue.columns]
        st.dataframe(queue[cols].head(10), use_container_width=True, hide_index=True)
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

elif page == "Predict a Claim":
    st.subheader("Assess a new claim")
    st.caption("Complete the fields below to obtain a model-assisted screening score.")
    feature_columns = [c for c in df.columns if c not in [TARGET, "PolicyNumber"]]
    with st.form("claim_prediction_form"):
        values = {}
        cols = st.columns(3)
        for idx, col in enumerate(feature_columns):
            series = df[col]
            with cols[idx % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    nums = pd.to_numeric(series, errors="coerce")
                    values[col] = st.number_input(col, value=float(nums.median()) if nums.notna().any() else 0.0, key=f"claim_{col}")
                else:
                    choices = series.dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                    values[col] = st.selectbox(col, choices, key=f"claim_{col}")
        submitted = st.form_submit_button("Run assessment", type="primary", use_container_width=True)
    if submitted:
        if not assets_ready: st.error("Add fraud_mlp.pt and preprocessor.pkl to enable prediction.")
        else:
            try:
                score = predict_one(pd.DataFrame([values]))
                a,b,c = st.columns(3)
                a.metric("Screening score", f"{score*100:.1f}/100")
                b.metric("Risk band", get_risk_band(score))
                c.metric("Predicted class", "Potential fraud" if score >= 0.5 else "Likely legitimate")
                st.progress(min(max(score,0.0),1.0))
                st.warning("This is an uncalibrated screening score, not proof of fraud. A human investigator should review the evidence.")
            except Exception as exc:
                st.error("Inference failed. Confirm the model weights, preprocessor and feature engineering match.")
                st.exception(exc)

elif page == "Fraud Risk Analysis":
    st.subheader("Fraud risk analysis")
    if predictions is None:
        st.error("Model inference is unavailable.")
        if prediction_error: st.code(prediction_error)
    else:
        a,b,c = st.columns(3)
        a.metric("High risk", f"{int((predictions['Risk']=='High risk').sum()):,}")
        b.metric("Medium risk", f"{int((predictions['Risk']=='Medium risk').sum()):,}")
        c.metric("Low risk", f"{int((predictions['Risk']=='Low risk').sum()):,}")
        st.markdown("#### Model score distribution")
        st.line_chart(predictions["Model score"].reset_index(drop=True).rename("Raw sigmoid score"))
        st.caption("Scores are not calibrated probabilities. This dashboard does not yet compute per-claim SHAP explanations.")
        band = st.selectbox("Filter by risk band", ["All", "High risk", "Medium risk", "Low risk"])
        view = df.copy()
        view["Prediction"] = predictions["Prediction"]
        view["Risk"] = predictions["Risk"]
        view["Model score"] = (predictions["Model score"]*100).round(1)
        if band != "All": view = view[view["Risk"] == band]
        st.dataframe(view.sort_values("Model score", ascending=False).head(100), use_container_width=True, hide_index=True)

elif page == "Claim Investigation":
    st.subheader("Claim investigation")
    st.caption("Browse the historical dataset and its recorded labels. Model predictions will appear here once prediction logging is implemented.")
    search = st.text_input("Search claims", placeholder="Search any visible field...")
    label_filter = st.selectbox("Historical label filter", ["All labels", "Fraud-labelled", "Legitimate-labelled"])
    view = df.copy()
    view["Historical label"] = np.where(historical == 1, "Fraud-labelled", "Legitimate-labelled")
    if label_filter != "All labels":
        view = view[view["Historical label"] == label_filter]
    if search.strip():
        searchable = view.astype(str).apply(
            lambda col: col.str.contains(search.strip(), case=False, na=False)
        ).any(axis=1)
        view = view[searchable]
    st.write(f"Showing {len(view):,} records")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered claims as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="claimcheck_filtered_claims.csv",
        mime="text/csv",
    )
    st.caption("The source file does not provide a verified claim ID for every record, so the dashboard does not invent one.")

elif page == "Model Performance":
    st.subheader("Model performance")
    st.write("These are the evaluation results recorded in the current notebook output. Re-run the notebook to refresh them after any training changes.")
    metrics = pd.DataFrame([
        {"Model": "Random Forest", "ROC-AUC": 0.8166, "Fraud recall": 0.01},
        {"Model": "Baseline XGBoost", "ROC-AUC": 0.8219, "Fraud recall": 0.33},
        {"Model": "PyTorch MLP", "ROC-AUC": 0.8310, "Fraud recall": 0.70},
    ])
    st.dataframe(
        metrics.style.format({
            "ROC-AUC": "{:.4f}", "Fraud recall": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("#### ROC-AUC comparison")
    st.bar_chart(metrics.set_index("Model")[["ROC-AUC"]])
    st.markdown("#### Fraud recall comparison")
    st.bar_chart(metrics.set_index("Model")[["Fraud recall"]])

    st.warning(
        "The current notebook outputs show substantial class imbalance and low fraud precision. "
        "For example, the MLP detects about 70% of fraud-labelled test cases, but its fraud precision "
        "is about 16%. These metrics are useful for demonstration and should be revalidated before deployment."
    )
    st.markdown("#### Confusion matrix")
    st.info(
        "A confusion matrix is not displayed yet because the notebook does not export the test labels "
        "and final predictions as a reusable artifact. We should save those from the evaluation cell "
        "rather than reconstructing or inventing counts."
    )

st.divider()
st.caption("ClaimCheck is an educational decision-support prototype. Flagged claims require human review; predictions are not proof of fraud.")
