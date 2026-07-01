"""
app.py
------
🏠 Household Electricity Consumption Predictor — Professional Streamlit App

Pages:
  🏠 Home
  📊 Dataset Overview
  📈 EDA Dashboard
  🤖 Model Comparison
  ⚡ Electricity Consumption Prediction
  📋 Model Performance
  📥 Download Prediction
  ℹ️ About

This app looks for these OPTIONAL files (missing ones are handled gracefully):
  - linear_regression_model.pkl          (your trained model — required for prediction)
  - data/household_electricity_data.csv  (your dataset — for Dataset/EDA pages)
  - outputs/model_metrics.csv            (for Model Comparison page)
  - outputs/metrics.json                 (for Model Performance page)
  - outputs/feature_importance.csv       (for Model Performance page)
"""

import json
import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/household_electricity_data.csv"
MODEL_PATH = "linear_regression_model.pkl"
METRICS_JSON = "outputs/metrics.json"
METRICS_CSV = "outputs/model_metrics.csv"
FI_CSV = "outputs/feature_importance.csv"

TARGET = "Monthly_Consumption_kWh"

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #1B4F72; margin-bottom: 0; }
    .sub-header { font-size: 1.05rem; color: #566573; margin-top: 0; }
    .metric-card {
        background: linear-gradient(135deg, #EBF5FB 0%, #D6EAF8 100%);
        border-radius: 12px; padding: 18px; border-left: 5px solid #2E86AB;
    }
    .footer { text-align: center; color: #808B96; padding-top: 2rem; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    data = pd.read_csv(DATA_PATH)
    # Align raw CSV column name with the TARGET name used throughout this app
    if "Monthly_Electricity_Consumption_kWh" in data.columns:
        data = data.rename(columns={"Monthly_Electricity_Consumption_kWh": TARGET})
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data
def load_metrics():
    metrics_json = json.load(open(METRICS_JSON)) if os.path.exists(METRICS_JSON) else None
    metrics_csv = pd.read_csv(METRICS_CSV) if os.path.exists(METRICS_CSV) else None
    fi_csv = pd.read_csv(FI_CSV) if os.path.exists(FI_CSV) else None
    return metrics_json, metrics_csv, fi_csv


df = load_dataset()
model = load_model()
metrics_json, metrics_df, fi_df = load_metrics()

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## ⚡ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home", "📊 Dataset Overview", "📈 EDA Dashboard", "🤖 Model Comparison",
        "⚡ Electricity Consumption Prediction", "📋 Model Performance",
        "📥 Download Prediction", "ℹ️ About",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
if model is not None:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.error("❌ Model not found (linear_regression_model.pkl)")
if metrics_json is not None:
    st.sidebar.info(f"🏆 Best model: **{metrics_json.get('best_model', 'N/A')}**")
st.sidebar.markdown("---")
st.sidebar.caption(f"📝 Predictions this session: {len(st.session_state.history)}")

# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏠 Household Electricity Consumption Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine-learning powered estimate of monthly household electricity usage (kWh)</p>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📦 Dataset Size", f"{len(df):,} rows" if df is not None else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        best_r2 = metrics_df["R2_Score"].max() if metrics_df is not None else None
        st.metric("🎯 Best R² Score", f"{best_r2:.3f}" if best_r2 is not None else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🤖 Models Compared", f"{len(metrics_df)}" if metrics_df is not None else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏆 Best Model", metrics_json.get("best_model", "N/A") if metrics_json else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["📖 Overview", "🧭 How to use", "🛠️ Tech Stack"])
    with tab1:
        st.write(
            "This application estimates a household's monthly electricity consumption "
            "(kWh) based on occupants, daily usage hours, house type, and appliances "
            "present. The best-performing trained model is used automatically for predictions."
        )
    with tab2:
        st.markdown(
            """
            1. **📊 Dataset Overview** — explore the raw data
            2. **📈 EDA Dashboard** — interactive charts and insights
            3. **🤖 Model Comparison** — compare algorithm performance
            4. **⚡ Prediction** — get a live estimate for a household
            5. **📋 Model Performance** — metrics and feature importance
            6. **📥 Download Prediction** — export your prediction history
            """
        )
    with tab3:
        st.markdown("- **Python** · **scikit-learn** · **pandas** · **Streamlit** · **Plotly**")

    st.markdown('<div class="footer">Built with ❤️ using Streamlit & scikit-learn</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: DATASET OVERVIEW
# ============================================================
elif page == "📊 Dataset Overview":
    st.markdown('<p class="main-header">📊 Dataset Overview</p>', unsafe_allow_html=True)

    if df is None:
        st.warning(f"⚠️ Dataset not found at `{DATA_PATH}`. Add your CSV there to enable this page.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))
        c4.metric("Avg. Consumption", f"{df[TARGET].mean():.1f} kWh" if TARGET in df.columns else "N/A")

        with st.expander("🔍 Preview Data", expanded=True):
            n_rows = st.slider("Rows to display", 5, 100, 10)
            st.dataframe(df.head(n_rows), use_container_width=True)

        with st.expander("📈 Statistical Summary"):
            st.dataframe(df.describe().T, use_container_width=True)

        with st.expander("🏷️ Column Data Types"):
            st.dataframe(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)}), use_container_width=True)

# ============================================================
# PAGE: EDA DASHBOARD
# ============================================================
elif page == "📈 EDA Dashboard":
    st.markdown('<p class="main-header">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if df is None:
        st.warning(f"⚠️ Dataset not found at `{DATA_PATH}`.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Distribution", "🔗 Correlations", "🏠 By House Type", "🔌 Appliance Impact"])

        with tab1:
            fig = px.histogram(df, x=TARGET, nbins=30, marginal="box", title="Distribution of Monthly Consumption")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            corr = df.select_dtypes(include="number").corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = px.box(df, x="Type_of_House", y=TARGET, color="Type_of_House", title="Consumption by House Type")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.scatter(df, x="Number_of_Occupants", y=TARGET, color="Type_of_House",
                               size="Daily_Usage_Hours_Avg", title="Occupants vs Consumption")
            st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            appliance_cols = [c for c in df.columns if c.startswith("Has_")]
            avg_impact = (
                df[appliance_cols + [TARGET]]
                .melt(id_vars=TARGET, var_name="Appliance", value_name="Present")
                .query("Present == 1")
                .groupby("Appliance")[TARGET].mean()
                .sort_values(ascending=True)
                .reset_index()
            )
            fig = px.bar(avg_impact, x=TARGET, y="Appliance", orientation="h",
                         title="Average Consumption When Appliance Is Present", color=TARGET)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================
elif page == "🤖 Model Comparison":
    st.markdown('<p class="main-header">🤖 Model Comparison</p>', unsafe_allow_html=True)

    if metrics_df is None:
        st.warning(f"⚠️ Metrics not found at `{METRICS_CSV}`.")
    else:
        best_row = metrics_df.sort_values("R2_Score", ascending=False).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Model", best_row["Model"])
        c2.metric("R² Score", f"{best_row['R2_Score']:.4f}")
        c3.metric("RMSE", f"{best_row['RMSE']:.2f} kWh")

        st.dataframe(metrics_df.sort_values("R2_Score", ascending=False), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(metrics_df.sort_values("R2_Score"), x="R2_Score", y="Model", orientation="h",
                         title="R² Score by Model (higher = better)", color="R2_Score")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(metrics_df.sort_values("RMSE", ascending=False), x="RMSE", y="Model", orientation="h",
                          title="RMSE by Model (lower = better)", color="RMSE")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE: PREDICTION
# ============================================================
elif page == "⚡ Electricity Consumption Prediction":
    st.markdown('<p class="main-header">⚡ Electricity Consumption Prediction</p>', unsafe_allow_html=True)
    st.write("Enter household details below to estimate monthly electricity consumption.")

    if model is None:
        st.error(f"❌ Model not found at `{MODEL_PATH}`.")
    else:
        with st.expander("🏠 Basic Household Information", expanded=True):
            c1, c2, c3 = st.columns(3)
            occupants = c1.number_input("Number of Occupants", 1, 20, 3)
            hours = c2.number_input("Daily Usage Hours (Avg)", 0.0, 24.0, 6.0)
            house_type = c3.selectbox("Type of House", ["Apartment", "Bungalow", "Single Room", "Self-Contained"])

        with st.expander("🔌 Appliances", expanded=True):
            ac1, ac2, ac3, ac4 = st.columns(4)
            has_fridge = ac1.checkbox("🧊 Fridge", value=True)
            has_tv = ac1.checkbox("📺 TV", value=True)
            has_iron = ac1.checkbox("👔 Electric Iron")
            has_fan = ac2.checkbox("🌀 Fan", value=True)
            has_ac = ac2.checkbox("❄️ Air Conditioner")
            has_washing = ac2.checkbox("🧺 Washing Machine")
            has_heater = ac3.checkbox("🚿 Water Heater")
            has_microwave = ac3.checkbox("🍲 Microwave")
            has_kettle = ac3.checkbox("☕ Electric Kettle")
            has_phone = ac4.checkbox("📱 Phone Charger", value=True)
            has_radio = ac4.checkbox("📻 Radio")
            has_bulbs = ac4.checkbox("💡 Multiple Lighting Bulbs", value=True)

        appliances = int(sum([has_fridge, has_tv, has_iron, has_fan, has_ac, has_washing,
                               has_heater, has_microwave, has_kettle, has_phone, has_radio, has_bulbs]))
        st.caption(f"🔢 Total appliances detected: **{appliances}**")

        if st.button("⚡ Predict Consumption", type="primary", use_container_width=True):
            input_df = pd.DataFrame([[
                occupants, appliances, hours,
                int(has_fridge), int(has_tv), int(has_iron), int(has_fan), int(has_ac),
                int(has_washing), int(has_heater), int(has_microwave), int(has_kettle),
                int(has_phone), int(has_radio), int(has_bulbs), house_type,
            ]], columns=[
                "Number_of_Occupants", "Number_of_Appliances", "Daily_Usage_Hours_Avg",
                "Has_Fridge", "Has_TV", "Has_Electric_Iron", "Has_Fan", "Has_Air_Conditioner",
                "Has_Washing_Machine", "Has_Water_Heater", "Has_Microwave", "Has_Electric_Kettle",
                "Has_Phone_Charger", "Has_Radio", "Has_Multiple_Lighting_Bulbs", "Type_of_House",
            ])

            try:
                raw_prediction = float(model.predict(input_df)[0])
                prediction = max(0.0, raw_prediction)  # consumption can't be negative
                if raw_prediction < 0:
                    st.warning(
                        f"⚠️ Model's raw output was {raw_prediction:.2f} kWh (negative). "
                        "This happens when inputs fall outside the range the model was "
                        "trained on. Displaying 0 kWh instead."
                    )
                st.success(f"⚡ Estimated Monthly Consumption: **{prediction:.2f} kWh**")

                c1, c2, c3 = st.columns(3)
                c1.metric("📅 Daily Avg.", f"{prediction / 30:.2f} kWh")
                c2.metric("📆 Monthly Total", f"{prediction:.2f} kWh")
                c3.metric("📈 Yearly Estimate", f"{prediction * 12:.0f} kWh")

                record = input_df.iloc[0].to_dict()
                record["Predicted_kWh"] = round(prediction, 2)
                record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append(record)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

        if st.session_state.history:
            st.markdown("### 🕓 Prediction History (this session)")
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "📋 Model Performance":
    st.markdown('<p class="main-header">📋 Model Performance</p>', unsafe_allow_html=True)

    if metrics_json is None:
        st.warning(f"⚠️ Metrics not found at `{METRICS_JSON}`.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model", metrics_json["best_model"])
        c2.metric("Train Rows", metrics_json.get("train_rows", "N/A"))
        c3.metric("Test Rows", metrics_json.get("test_rows", "N/A"))

        st.dataframe(metrics_df, use_container_width=True)

        if fi_df is not None:
            st.markdown("### 🌟 Feature Importance")
            fig = px.bar(fi_df.head(15).sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", title="Top Feature Importances", color="Importance")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: DOWNLOAD PREDICTION
# ============================================================
elif page == "📥 Download Prediction":
    st.markdown('<p class="main-header">📥 Download Prediction</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("ℹ️ No predictions yet. Go to **⚡ Electricity Consumption Prediction** to make one first.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.markdown(f"### 🗂️ {len(hist_df)} Prediction(s) in This Session")
        st.dataframe(hist_df, use_container_width=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prediction History as CSV", data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", type="primary", use_container_width=True,
        )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown(
        """
        ### 🏠 Household Electricity Consumption Predictor

        Estimates a household's monthly electricity consumption using a
        trained regression model based on occupancy, usage habits, house
        type, and appliance ownership.

        **Tech Stack:** Python, scikit-learn, pandas, Streamlit, Plotly

        **License:** MIT
        """
    )
    st.markdown('<div class="footer">© 2026 Household Electricity Consumption Predictor</div>', unsafe_allow_html=True)