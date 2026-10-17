from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ClaimCheck | Jury Demo", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.2rem;max-width:1450px}
[data-testid="stMetric"] {padding:18px;border-radius:14px;background:#172233;border:1px solid #344458}
.hero {padding:24px 30px;border-radius:18px;background:linear-gradient(115deg,#142a42,#143e45);color:white;margin-bottom:22px}
.hero h1 {color:white;margin:0}
.hero p {color:#c9d7e4;margin:8px 0 0}
</style>
""", unsafe_allow_html=True)

DATA = Path(__file__).resolve().parent / "fraud_oracle.csv"

@st.cache_data(show_spinner=False)
def get_data(path, mtime):
    return pd.read_csv(path)

if not DATA.exists():
    st.error("fraud_oracle.csv is missing from the repository.")
    st.stop()
try:
    df = get_data(str(DATA), DATA.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

target = "FraudFound_P"
if target not in df.columns:
    st.error("Dataset does not contain FraudFound_P.")
    st.stop()

labels = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
fraud = int((labels == 1).sum())
legit = int((labels == 0).sum())

with st.sidebar:
    st.title("🛡️ ClaimCheck")
    st.caption("Insurance fraud intelligence")
    page = st.radio("Explore", ["Overview", "Claim Explorer", "Fraud Patterns", "Model Performance", "AI Prediction (Next Stage)"])
    st.divider()
    st.success("Demo dashboard ready")
    st.caption("Historical dataset analytics · No live model inference")
    st.caption("Source: fraud_oracle.csv")

st.markdown('<div class="hero"><h1>🛡️ ClaimCheck</h1><p>Insurance claim fraud detection · Interactive jury demonstration</p></div>', unsafe_allow_html=True)
st.info("**Demo mode:** This dashboard uses actual historical dataset labels. Model-based predictions and risk scores are not displayed until the saved preprocessing pipeline is compatible.")

if page == "Overview":
    st.subheader("Claims intelligence overview")
    a,b,c,d = st.columns(4)
    a.metric("Total claims", f"{len(df):,}")
    b.metric("Historically fraudulent", f"{fraud:,}")
    c.metric("Historically legitimate", f"{legit:,}")
    d.metric("Historical fraud rate", f"{fraud/len(df):.2%}" if len(df) else "0%")
    left,right = st.columns(2)
    with left:
        st.markdown("#### Historical claim outcomes")
        st.bar_chart(pd.Series({"Legitimate":legit,"Fraudulent":fraud}),horizontal=True)
    with right:
        st.markdown("#### Claims by vehicle make")
        if "Make" in df:
            st.bar_chart(df["Make"].fillna("Unknown").value_counts().head(10),horizontal=True)
    left,right = st.columns(2)
    with left:
        st.markdown("#### Accident area and historical fraud")
        if "AccidentArea" in df:
            st.bar_chart(pd.crosstab(df["AccidentArea"],labels).rename(columns={0:"Legitimate",1:"Fraudulent"}))
    with right:
        st.markdown("#### Vehicle price categories")
        if "VehiclePrice" in df:
            st.bar_chart(df["VehiclePrice"].fillna("Unknown").value_counts(),horizontal=True)
    st.markdown("#### Dataset snapshot")
    st.dataframe(df.head(12),use_container_width=True,hide_index=True)

elif page == "Claim Explorer":
    st.subheader("Claim investigation workspace")
    view = df.copy()
    view.insert(0,"Demo record ID",[f"CL-{i+1:05d}" for i in range(len(df))])
    view["Historical outcome"] = labels.map({0:"Legitimate",1:"Fraud-labelled"})
    c1,c2,c3 = st.columns(3)
    with c1: search = st.text_input("Search records",placeholder="Vehicle, area, ID, policy…")
    with c2: outcome = st.selectbox("Historical outcome",["All","Fraud-labelled","Legitimate"])
    with c3:
        makes = sorted(df["Make"].dropna().astype(str).unique().tolist()) if "Make" in df else []
        make = st.selectbox("Vehicle make",["All"]+makes)
    if outcome!="All": view=view[view["Historical outcome"]==outcome]
    if make!="All": view=view[view["Make"]==make]
    if search.strip():
        mask=view.astype(str).apply(lambda x:x.str.contains(search.strip(),case=False,regex=False,na=False)).any(axis=1)
        view=view[mask]
    st.metric("Matching claims",f"{len(view):,}")
    st.dataframe(view.head(500),use_container_width=True,hide_index=True)
    st.download_button("Export filtered claims",view.to_csv(index=False).encode(),"claimcheck_filtered.csv","text/csv")
    if len(view):
        selected=st.selectbox("Inspect a record",view["Demo record ID"].head(500).tolist())
        row=view.loc[view["Demo record ID"]==selected].iloc[0]
        st.markdown(f"#### Claim record: {selected}")
        st.write(f"**Historical outcome:** {row['Historical outcome']}")
        st.dataframe(pd.DataFrame({"Field":row.index.astype(str),"Value":row.astype(str).values}),hide_index=True,use_container_width=True)
    st.caption("Demo record IDs are generated from dataset row order, not official insurance claim identifiers.")

elif page == "Fraud Patterns":
    st.subheader("Fraud patterns in historical claims")
    field_options=[c for c in ["Make","AccidentArea","Fault","VehiclePrice","AgeOfVehicle","PolicyType","BasePolicy","Sex","MaritalStatus"] if c in df]
    if field_options:
        field=st.selectbox("Compare fraud outcomes by",field_options)
        summary=pd.crosstab(df[field].fillna("Unknown"),labels).rename(columns={0:"Legitimate",1:"Fraudulent"})
        st.bar_chart(summary)
        rate=df.assign(_fraud=labels).groupby(field,dropna=False)["_fraud"].agg(["size","mean"]).reset_index()
        rate.columns=[field,"Claims","Historical fraud rate"]
        rate["Historical fraud rate"]=(rate["Historical fraud rate"]*100).round(2)
        st.dataframe(rate.sort_values("Historical fraud rate",ascending=False),use_container_width=True,hide_index=True)
    st.caption("These patterns describe the historical dataset; association does not establish causation.")

elif page == "Model Performance":
    st.subheader("Model evaluation")
    st.caption("Figures recorded in the training notebook; not recomputed during this demonstration.")
    metrics=pd.DataFrame([
        {"Model":"Random Forest","ROC-AUC":0.8166,"Fraud recall":0.01},
        {"Model":"XGBoost baseline","ROC-AUC":0.8219,"Fraud recall":0.33},
        {"Model":"PyTorch MLP","ROC-AUC":0.8310,"Fraud recall":0.70},
    ])
    st.dataframe(metrics,use_container_width=True,hide_index=True)
    left,right=st.columns(2)
    with left:
        st.markdown("#### ROC-AUC comparison")
        st.bar_chart(metrics.set_index("Model")["ROC-AUC"])
    with right:
        st.markdown("#### Fraud recall comparison")
        st.bar_chart(metrics.set_index("Model")["Fraud recall"])
    st.warning("Class imbalance and low fraud precision are important limitations. Evaluation metrics should be verified against the saved test predictions before production use.")

else:
    st.subheader("AI claim screening · integration in progress")
    st.write("The trained PyTorch MLP and fitted preprocessing pipeline are available as project artifacts, but the deployed environment currently has a scikit-learn serialization compatibility issue.")
    st.info("For today's demonstration, use Overview, Claim Explorer, Fraud Patterns, and Model Performance. Live risk scoring will be enabled after the preprocessor compatibility issue is resolved.")
    st.markdown("#### Intended workflow")
    st.code("Claim details → Feature engineering → Preprocessor → PyTorch MLP → Risk score → Human investigation")

st.divider()
st.caption("ClaimCheck is an educational prototype. Historical fraud labels are not model predictions; suspicious claims require human review.")
