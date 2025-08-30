
import os
import io
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

st.set_page_config(page_title="🏥 NHI in SA — Dashboard", page_icon="🏥", layout="wide")

@st.cache_data
def read_csv(file_or_buffer):
    return pd.read_csv(file_or_buffer)

def kpi_card(col, label, value):
    col.metric(label, value)

def safe_parse_dates(df, date_cols=("date","appointment_date","visit_date","created_at")):
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def build_sample_data(n_patients=500, seed=42):
    rng = np.random.default_rng(seed)
    provinces = ["Gauteng","KwaZulu-Natal","Western Cape","Eastern Cape","Limpopo","Mpumalanga","North West","Free State","Northern Cape"]
    sexes = ["Male","Female"]
    specialties = ["GP","Pediatrics","Cardiology","Orthopedics","OBGYN"]
    facility_types = ["Clinic","District Hospital","Regional Hospital","Private Hospital"]

    patients = pd.DataFrame({
        "patient_id": np.arange(1, n_patients+1),
        "age": rng.integers(0, 90, size=n_patients),
        "sex": rng.choice(sexes, size=n_patients),
        "province": rng.choice(provinces, size=n_patients),
        "income_bracket": rng.choice(["Low","Middle","High"], size=n_patients, p=[0.55, 0.35, 0.10]),
    })

    n_docs = 120
    doctors = pd.DataFrame({
        "doctor_id": np.arange(1, n_docs+1),
        "specialty": rng.choice(specialties, size=n_docs),
        "facility_type": rng.choice(facility_types, size=n_docs),
        "province": rng.choice(provinces, size=n_docs),
    })

    n_appts = n_patients * rng.integers(1, 6)
    appt_dates = pd.date_range("2023-01-01", "2025-08-01", freq="D")
    appointments = pd.DataFrame({
        "appointment_id": np.arange(1, n_appts+1),
        "patient_id": rng.integers(1, n_patients+1, size=n_appts),
        "doctor_id": rng.integers(1, n_docs+1, size=n_appts),
        "appointment_date": rng.choice(appt_dates, size=n_appts),
        "status": rng.choice(["attended", "no_show"], size=n_appts, p=[0.85, 0.15]),
    })

    base_cost = {
        "Clinic": (150, 400),
        "District Hospital": (300, 1200),
        "Regional Hospital": (600, 2500),
        "Private Hospital": (1200, 7000),
    }

    merged = appointments.merge(doctors[["doctor_id","facility_type"]], on="doctor_id", how="left")
    lo, hi = zip(*[base_cost.get(ft, (200, 2000)) for ft in merged["facility_type"].fillna("Clinic")])
    amounts = np.random.default_rng(seed+1).uniform(np.array(lo), np.array(hi))
    billing = merged[["appointment_id"]].copy()
    billing["amount"] = amounts.round(2)

    return patients, appointments, billing, doctors

def merge_tables(patients, appointments, billing, doctors):
    df = appointments.merge(patients, on="patient_id", how="left") \
                     .merge(doctors, on="doctor_id", how="left") \
                     .merge(billing, on="appointment_id", how="left")
    df = safe_parse_dates(df)
    return df

def detect_task_type(y: pd.Series):
    if y.dtype.kind in "ifu":
        uniq = np.unique(y.dropna())
        if len(uniq) <= 10 and set(uniq).issubset({0,1}):
            return "classification"
        return "regression"
    else:
        return "classification"

def summarize_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, average="weighted")
    rec = recall_score(y_true, y_pred, zero_division=0, average="weighted")
    f1 = f1_score(y_true, y_pred, zero_division=0, average="weighted")
    return acc, prec, rec, f1

def summarize_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

st.title("🏥 NHI in South Africa — Analytics Dashboard")
st.caption("Streamlit app with **2 tabs**: Explore & Model. Upload your data or use the synthetic sample.")

# Sidebar — data input
st.sidebar.title("📥 Data Input")
st.sidebar.caption("Upload your 4 related tables **or** generate a sample dataset.")

with st.sidebar.expander("Upload CSV files", expanded=False):
    up_patients = st.file_uploader("patients.csv", type=["csv"], key="patients")
    up_appointments = st.file_uploader("appointments.csv", type=["csv"], key="appointments")
    up_billing = st.file_uploader("billing.csv", type=["csv"], key="billing")
    up_doctors = st.file_uploader("doctors.csv", type=["csv"], key="doctors")

use_sample = st.sidebar.checkbox("Use sample data (synthetic)", value=True)
model_path = st.sidebar.text_input("Model file name", value="model.pkl")

# Load data
if use_sample:
    patients_df, appointments_df, billing_df, doctors_df = build_sample_data()
else:
    if not (up_patients and up_appointments and up_billing and up_doctors):
        st.warning("Upload all four CSVs or tick **Use sample data (synthetic)** in the sidebar.")
        st.stop()
    patients_df = read_csv(up_patients)
    appointments_df = read_csv(up_appointments)
    billing_df = read_csv(up_billing)
    doctors_df = read_csv(up_doctors)

# Quick validation (soft)
required_cols = {
    "patients": ["patient_id", "age", "sex", "province"],
    "appointments": ["appointment_id", "patient_id", "doctor_id", "appointment_date"],
    "billing": ["appointment_id", "amount"],
    "doctors": ["doctor_id", "specialty", "province", "facility_type"],
}
missing_msgs = []
for name, df_, cols in [
    ("patients", patients_df, required_cols["patients"]),
    ("appointments", appointments_df, required_cols["appointments"]),
    ("billing", billing_df, required_cols["billing"]),
    ("doctors", doctors_df, required_cols["doctors"]),
]:
    miss = [c for c in cols if c not in df_.columns]
    if miss:
        missing_msgs.append(f"- **{name}** missing: {', '.join(miss)}")
if missing_msgs:
    st.warning("Some expected columns are missing (the app will try to continue):\n" + "\n".join(missing_msgs))

df = merge_tables(patients_df, appointments_df, billing_df, doctors_df)

# Tabs
explore_tab, model_tab = st.tabs(["📊 Explore", "🤖 Model"])

# Explore Tab
with explore_tab:
    st.subheader("Data Overview")
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Patients", f"{df['patient_id'].nunique():,}" if "patient_id" in df.columns else "N/A")
    kpi_card(c2, "Visits", f"{df['appointment_id'].nunique():,}" if "appointment_id" in df.columns else "N/A")
    total_spend = df["amount"].sum() if "amount" in df.columns else 0.0
    kpi_card(c3, "Total Spend (R)", f"{total_spend:,.2f}")

    st.markdown("#### Filters")
    colf1, colf2, colf3 = st.columns(3)
    province_opts = sorted(df["province"].dropna().astype(str).unique()) if "province" in df.columns else []
    province_sel = colf1.multiselect("Province", province_opts, default=province_opts[:3] if province_opts else [])
    if "appointment_date" in df.columns:
        year_opts = sorted(set(d.year for d in df["appointment_date"].dropna()))
    else:
        year_opts = []
    year_sel = colf2.multiselect("Year", year_opts, default=year_opts)
    status_opts = sorted(df["status"].dropna().astype(str).unique()) if "status" in df.columns else []
    status_sel = colf3.multiselect("Status", status_opts, default=status_opts)

    dff = df.copy()
    if province_sel and "province" in dff.columns:
        dff = dff[dff["province"].astype(str).isin(province_sel)]
    if year_sel and "appointment_date" in dff.columns:
        dff = dff[dff["appointment_date"].dt.year.isin(year_sel)]
    if status_sel and "status" in dff.columns:
        dff = dff[dff["status"].astype(str).isin(status_sel)]

    st.markdown("#### Visits by Province")
    if "province" in dff.columns and "appointment_id" in dff.columns:
        by_prov = dff.groupby("province")["appointment_id"].nunique().reset_index(name="visits")
        fig1 = px.bar(by_prov, x="province", y="visits")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Need `province` and `appointment_id` columns to plot by province.")

    st.markdown("#### Monthly Visit Trend")
    if "appointment_date" in dff.columns and "appointment_id" in dff.columns:
        dff["year_month"] = dff["appointment_date"].dt.to_period("M").astype(str)
        trend = dff.groupby("year_month")["appointment_id"].nunique().reset_index(name="visits")
        fig2 = px.line(trend, x="year_month", y="visits")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Need `appointment_date` and `appointment_id` to plot monthly trend.")

    st.markdown("#### Data Preview")
    st.dataframe(dff.head(50))

# Model Tab
with model_tab:
    st.subheader("Train a Simple ML Model")

    cols = df.columns.tolist()
    target_col = st.selectbox("Target column (what to predict)", options=["<auto-create: high_utilizer>"] + cols, index=0)

    work = df.copy()

    if target_col == "<auto-create: high_utilizer>":
        if "appointment_date" in work.columns:
            work["year"] = work["appointment_date"].dt.year
        else:
            work["year"] = 2025
        visits = work.groupby(["patient_id","year"])["appointment_id"].transform("count") if "appointment_id" in work.columns else pd.Series([0]*len(work))
        threshold = int(np.percentile(visits, 75)) if len(visits)>0 else 3
        work["high_utilizer"] = (visits >= max(threshold, 3)).astype(int)
        y = work["high_utilizer"]
        task_type = "classification"
        st.caption(f"Auto-created target **high_utilizer** (≥ {max(threshold,3)} visits/year ⇒ 1)")
    else:
        y = work[target_col]
        task_type = "classification" if y.dtype.kind not in "ifu" else ("classification" if set(np.unique(y.dropna())).issubset({0,1}) else "regression")

    drop_cols = {"appointment_id","patient_id","doctor_id"}
    if task_type == "classification" and target_col == "<auto-create: high_utilizer>":
        drop_cols.add("high_utilizer")
    X = work.drop(columns=[c for c in drop_cols if c in work.columns], errors="ignore")
    if target_col != "<auto-create: high_utilizer>" and target_col in X.columns:
        X = X.drop(columns=[target_col])

    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    st.markdown("#### Algorithm")
    if task_type == "classification":
        algo = st.selectbox("Choose classifier", ["RandomForestClassifier", "LogisticRegression"], index=0)
    else:
        algo = st.selectbox("Choose regressor", ["RandomForestRegressor", "LinearRegression"], index=0)

    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training..."):
            try:
                if task_type == "classification":
                    if algo == "RandomForestClassifier":
                        model = RandomForestClassifier(n_estimators=300, random_state=42)
                    else:
                        model = LogisticRegression(max_iter=300)
                else:
                    if algo == "RandomForestRegressor":
                        model = RandomForestRegressor(n_estimators=300, random_state=42)
                    else:
                        model = LinearRegression()

                pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if task_type=="classification" and y.nunique() > 1 else None
                )
                pipe.fit(X_train, y_train)

                if task_type == "classification":
                    y_pred = pipe.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0, average="weighted")
                    rec = recall_score(y_test, y_pred, zero_division=0, average="weighted")
                    f1 = f1_score(y_test, y_pred, zero_division=0, average="weighted")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.3f}")
                    m2.metric("Precision (weighted)", f"{prec:.3f}")
                    m3.metric("Recall (weighted)", f"{rec:.3f}")
                    m4.metric("F1-score (weighted)", f"{f1:.3f}")

                    cm = confusion_matrix(y_test, y_pred)
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(pd.DataFrame(cm))

                else:
                    y_pred = pipe.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    r2 = r2_score(y_test, y_pred)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:,.3f}")
                    m2.metric("RMSE", f"{rmse:,.3f}")
                    m3.metric("R²", f"{r2:,.3f}")

                joblib.dump(pipe, model_path)
                st.success(f"Model saved to **{model_path}**")
                with open(model_path, "rb") as f:
                    st.download_button("Download model.pkl", f, file_name=model_path, use_container_width=True)

            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")
    st.markdown("#### Quick Single Prediction")
    st.caption("Load the saved model and enter a single row of inputs to predict.")
    if os.path.exists(model_path):
        if st.button("Load Saved Model", use_container_width=True):
            try:
                pipe = joblib.load(model_path)
                st.success("Model loaded.")

                inputs = {}
                num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
                cat_cols = [c for c in X.columns if c not in num_cols]

                with st.form("single_pred"):
                    for c in num_cols:
                        val = float(X[c].median()) if len(X[c]) else 0.0
                        inputs[c] = st.number_input(c, value=val)
                    for c in cat_cols:
                        uniq = sorted(X[c].astype(str).dropna().unique().tolist())[:100]
                        inputs[c] = st.selectbox(c, uniq or ["Unknown"], index=0)
                    submitted = st.form_submit_button("Predict")
                if submitted:
                    row = pd.DataFrame([inputs])
                    try:
                        pred = pipe.predict(row)[0]
                        st.success(f"Prediction: **{pred}**")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            except Exception as e:
                st.error(f"Load failed: {e}")
    else:
        st.info("No saved model found yet. Train a model first.")
