import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="NHI Healthcare Demand Dashboard", layout="wide")
st.title("📊 NHI Healthcare Demand Prediction Dashboard")
st.caption("Assessment 4 – Task One")

# ---------- Load data with friendly errors ----------
def safe_read_csv(name):
    p = Path(name)
    if not p.exists():
        st.error(f"Missing required file: `{name}`. Upload it to the repo root.")
        st.stop()
    return pd.read_csv(p)

demand_df = safe_read_csv("demand_df.csv")
master_df = safe_read_csv("master_df.csv")
reg_results_df = safe_read_csv("reg_results.csv")
class_results_df = safe_read_csv("class_results.csv")

# Optional files
fi_df = pd.read_csv("rf_feature_importance.csv") if Path("rf_feature_importance.csv").exists() else None
cm_df = pd.read_csv("confusion_matrix.csv", index_col=0) if Path("confusion_matrix.csv").exists() else None

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Data & Insights", "Model Results"])

with tab1:
    st.subheader("Quick Metrics")
    total_appts = int(demand_df["appointment_count"].sum())
    unique_patients = master_df[[c for c in master_df.columns if c.startswith("patient_id") or c=="patient_id"]].nunique().max()
    weekend_share = master_df["is_weekend"].mean() if "is_weekend" in master_df.columns else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Appointments", f"{total_appts}")
    col2.metric("Unique Patients", f"{int(unique_patients)}")
    col3.metric("Weekend Share", f"{weekend_share:.0%}")

    st.markdown("### Monthly Appointment Demand")
    monthly = demand_df.groupby("appointment_month")["appointment_count"].sum()
    st.bar_chart(monthly)

    st.markdown("### Appointments by Treatment Type")
    if "treatment_type" in demand_df.columns:
        treatment = demand_df.groupby("treatment_type")["appointment_count"].sum().sort_values(ascending=False)
        st.bar_chart(treatment)
    else:
        st.info("`treatment_type` not found in demand_df.")

    st.markdown("### Appointments by Age Group")
    if "age_group" in master_df.columns:
        st.bar_chart(master_df["age_group"].value_counts())
    else:
        st.info("`age_group` not found in master_df.")

    st.markdown("### Weekend vs Weekday")
    if "is_weekend" in master_df.columns:
        wkd = master_df["is_weekend"].map({0: "Weekday", 1: "Weekend"}).value_counts()
        st.bar_chart(wkd)
    else:
        st.info("`is_weekend` not found in master_df.")

with tab2:
    st.subheader("Regression Models")
    if {"Model","MAE","RMSE","R²","CV R²"}.issubset(reg_results_df.columns):
        st.dataframe(reg_results_df.sort_values("R²", ascending=False), use_container_width=True)

        st.write("**R² Comparison**")
        fig, ax = plt.subplots()
        ax.bar(reg_results_df["Model"], reg_results_df["R²"])
        ax.set_ylabel("R² Score")
        ax.set_title("Regression Model Comparison")
        plt.xticks(rotation=15, ha="right")
        st.pyplot(fig)
    else:
        st.error("reg_results.csv is missing expected columns: Model, MAE, RMSE, R², CV R²")

    st.markdown("---")
    st.subheader("Classification Models")
    if {"Model","Accuracy","Precision","Recall","F1-Score"}.issubset(class_results_df.columns):
        st.dataframe(class_results_df.sort_values("Accuracy", ascending=False), use_container_width=True)

        st.write("**Accuracy Comparison**")
        fig2, ax2 = plt.subplots()
        ax2.bar(class_results_df["Model"], class_results_df["Accuracy"])
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Classification Model Comparison")
        plt.xticks(rotation=15, ha="right")
        st.pyplot(fig2)
    else:
        st.error("class_results.csv is missing expected columns: Model, Accuracy, Precision, Recall, F1-Score")

    # Optional polish: Feature importance and Confusion Matrix
    if fi_df is not None and "feature" in fi_df.columns and "importance" in fi_df.columns:
        st.markdown("### Random Forest Classifier – Feature Importance")
        topn = fi_df.head(10)
        fig3, ax3 = plt.subplots()
        ax3.barh(topn["feature"][::-1], topn["importance"][::-1])
        ax3.set_xlabel("Importance")
        ax3.set_title("Top 10 Features")
        st.pyplot(fig3)

    if cm_df is not None:
        st.markdown("### Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", ax=ax4)
        ax4.set_title("Confusion Matrix (RF Classifier)")
        st.pyplot(fig4)
