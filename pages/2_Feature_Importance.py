import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Feature Importance", page_icon="🧠", layout="wide")

st.title("🧠 Model Explainability")
st.markdown("Understand which pollutants drive AQI predictions the most.")

@st.cache_resource
def load_model():
    if not os.path.exists("aqi_model.pkl"):
        return None
    return joblib.load("aqi_model.pkl")

model = load_model()

if model is None:
    st.error("❌ `aqi_model.pkl` not found. Train the model first.")
    st.stop()

FEATURES = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]

# ---------- Feature Importance ----------
st.subheader("📊 Feature Importance (from Model)")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis",
                 title="Feature Importance (Tree-based Model)")
    st.plotly_chart(fig)

    st.markdown("**Key Takeaways:**")
    top3 = fi_df.nlargest(3, "Importance")["Feature"].tolist()
    st.success(f"Top 3 most influential pollutants: **{', '.join(top3)}**")

elif hasattr(model, "coef_"):
    coefs = model.coef_
    coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": coefs})
    coef_df = coef_df.sort_values("Coefficient", ascending=True)

    fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                 color="Coefficient", color_continuous_scale="RdBu",
                 title="Model Coefficients (Linear Model)")
    st.plotly_chart(fig)
else:
    st.warning("This model type does not expose feature importances directly.")

st.markdown("---")

# ---------- Model Info ----------
st.subheader("🤖 Model Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Model Type:** `{type(model).__name__}`")
    if hasattr(model, "n_estimators"):
        st.markdown(f"**Number of Estimators:** {model.n_estimators}")
    if hasattr(model, "max_depth"):
        st.markdown(f"**Max Depth:** {model.max_depth}")
    if hasattr(model, "random_state"):
        st.markdown(f"**Random State:** {model.random_state}")

with col2:
    st.markdown("**Input Features:**")
    for f in FEATURES:
        st.markdown(f"- {f}")

st.markdown("---")

# ---------- Manual What-If Analysis ----------
st.subheader("🔬 What-If Analysis")
st.markdown("Slide a pollutant value and see how it affects the predicted AQI.")

what_if_feature = st.selectbox("Select Pollutant to Vary", FEATURES)
what_if_range   = st.slider(f"{what_if_feature} range", 0.0, 500.0, (0.0, 200.0))

base_values = {f: 30.0 for f in FEATURES}

sweep_vals = np.linspace(what_if_range[0], what_if_range[1], 100)
preds = []
for v in sweep_vals:
    row = base_values.copy()
    row[what_if_feature] = v
    preds.append(model.predict(pd.DataFrame([row]))[0])

sweep_df = pd.DataFrame({"Pollutant Value": sweep_vals, "Predicted AQI": preds})
fig2 = px.line(sweep_df, x="Pollutant Value", y="Predicted AQI",
               title=f"Effect of {what_if_feature} on AQI (all other pollutants held at 30)",
               color_discrete_sequence=["#e74c3c"])
fig2.add_hline(y=100, line_dash="dot", annotation_text="Moderate threshold")
fig2.add_hline(y=200, line_dash="dot", line_color="red", annotation_text="Poor threshold")
st.plotly_chart(fig2)
