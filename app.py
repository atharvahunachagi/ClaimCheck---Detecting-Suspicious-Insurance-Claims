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
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
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
        ["Overview & Analytics", "Predict a Claim", "Claim Investigation", "Model Performance"],
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

if page == "Overview & Analytics":
    st.subheader("Dashboard overview")
    st.caption("Dataset statistics below are calculated from the repository CSV. Prediction KPIs remain unavailable until the trained model artifacts are added.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims in dataset", f"{len(df):,}")
    c2.metric("High-risk predictions", "—", help="Requires the saved model and preprocessor.")
    c3.metric("Low-risk predictions", "—", help="Requires the saved model and preprocessor.")
    c4.metric("Model status", "Ready" if assets_ready else "Not connected")

    st.info(
        f"Historical labels: {fraud_count:,} fraud-labelled claims ({fraud_pct:.2f}%) "
        f"and {legit_count:,} legitimate-labelled claims. These are dataset labels, "
        "not predictions made by the dashboard."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Historical label distribution")
        label_data = pd.DataFrame({
            "Label": ["Legitimate", "Fraudulent"],
            "Claims": [legit_count, fraud_count],
        }).set_index("Label")
        st.bar_chart(label_data)
    with right:
        st.markdown("#### Claims by vehicle make")
        if "Make" in df.columns:
            make_counts = df["Make"].fillna("Unknown").value_counts().head(12)
            st.bar_chart(make_counts)
        else:
            st.write("Vehicle make is not available in this dataset.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Historical fraud labels by accident area")
        if "AccidentArea" in df.columns:
            area = pd.crosstab(df["AccidentArea"].fillna("Unknown"), historical)
            area = area.rename(columns={0: "Legitimate label", 1: "Fraud label"})
            st.bar_chart(area)
        else:
            st.write("Accident-area field is not available.")
    with right:
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.caption("Historical labels describe the source dataset. They must not be presented as model-generated risk predictions.")

elif page == "Predict a Claim":
    st.subheader("Predict fraud risk for a claim")
    st.write("Enter the claim and policy information. The form uses the feature fields available in the training dataset.")

    if not assets_ready:
        st.warning(
            "Live prediction is currently unavailable because the repository does not contain both "
            "'fraud_mlp.pt' and 'preprocessor.pkl'. The form can be reviewed, but no placeholder "
            "or fabricated prediction will be shown."
        )

    feature_columns = [c for c in df.columns if c not in [TARGET, "PolicyNumber"]]
    with st.form("claim_prediction_form"):
        st.markdown("##### Claim details")
        values = {}
        columns = st.columns(3)
        for idx, col in enumerate(feature_columns):
            series = df[col]
            with columns[idx % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    median = pd.to_numeric(series, errors="coerce").median()
                    default = float(median) if pd.notna(median) else 0.0
                    values[col] = st.number_input(
                        col, value=default, key=f"claim_{col}"
                    )
                else:
                    choices = series.dropna().astype(str).value_counts().index.tolist()
                    if not choices:
                        choices = ["Unknown"]
                    values[col] = st.selectbox(col, choices, key=f"claim_{col}")
        submitted = st.form_submit_button("Analyze claim", type="primary", use_container_width=True)

    if submitted:
        if not assets_ready:
            st.error("Prediction not run: add the trained model and preprocessor artifacts first.")
        else:
            try:
                input_frame = pd.DataFrame([values])
                input_frame = prepare_features(input_frame)
                preprocessor = load_preprocessor(
                    str(PREPROCESSOR_PATH), PREPROCESSOR_PATH.stat().st_mtime
                )
                transformed = preprocessor.transform(input_frame)
                transformed = np.asarray(transformed, dtype=np.float32)
                model = FraudMLP(input_dim=transformed.shape[1])
                state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(transformed, dtype=torch.float32))
                    probability = float(torch.sigmoid(logits).item())

                band = get_risk_band(probability)
                score = round(probability * 100)
                a, b, c = st.columns(3)
                a.metric("Predicted class", "Potential fraud" if probability >= 0.5 else "Likely legitimate")
                b.metric("Model score", f"{score}/100")
                c.metric("Risk band", band)
                st.progress(min(max(probability, 0.0), 1.0))
                st.warning(
                    "This is a model score, not a calibrated probability or proof of fraud. "
                    "Review the claim and supporting evidence before making a decision."
                )
            except Exception as exc:
                st.error("Prediction failed. The saved model, preprocessor, and feature engineering must match the notebook.")
                st.exception(exc)

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
        {"Model": "Random Forest", "ROC-AUC": 0.8166, "Fraud precision": 1.00, "Fraud recall": 0.01, "Fraud F1": 0.01},
        {"Model": "Baseline XGBoost", "ROC-AUC": 0.8219, "Fraud precision": 0.19, "Fraud recall": 0.33, "Fraud F1": 0.24},
        {"Model": "PyTorch MLP", "ROC-AUC": 0.8310, "Fraud precision": 0.16, "Fraud recall": 0.70, "Fraud F1": 0.26},
    ])
    st.dataframe(
        metrics.style.format({
            "ROC-AUC": "{:.4f}", "Fraud precision": "{:.2%}",
            "Fraud recall": "{:.2%}", "Fraud F1": "{:.2%}",
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
