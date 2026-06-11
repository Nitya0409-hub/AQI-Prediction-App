import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance")
st.markdown("Evaluate how well the trained model predicts AQI.")

@st.cache_resource
def load_model():
    if not os.path.exists("aqi_model.pkl"):
        return None
    return joblib.load("aqi_model.pkl")

@st.cache_data
def load_data():
    for path in ["AQI_CLEANED.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path), path
    return None, None

model = load_model()
df, source = load_data()

FEATURES = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]
TARGET = "AQI"

if model is None:
    st.error("❌ `aqi_model.pkl` not found.")
    st.stop()

if df is None:
    st.error("❌ Dataset not found. Place `AQI_CLEANED.csv` in the project folder.")
    st.stop()

# Prepare data
avail_features = [f for f in FEATURES if f in df.columns]
if TARGET not in df.columns:
    st.error("❌ 'AQI' column not found in dataset.")
    st.stop()

data = df[avail_features + [TARGET]].dropna()
X = data[avail_features]
y = data[TARGET]

# Pad missing features with 0
for f in FEATURES:
    if f not in X.columns:
        X[f] = 0.0
X = X[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# ---------- Metrics ----------
st.subheader("📊 Evaluation Metrics (Test Set — 20%)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² Score",  f"{r2:.4f}",   help="Closer to 1.0 is better")
c2.metric("MAE",       f"{mae:.2f}",  help="Mean Absolute Error — lower is better")
c3.metric("RMSE",      f"{rmse:.2f}", help="Root Mean Squared Error — lower is better")
c4.metric("Test Samples", f"{len(y_test):,}")

st.markdown("---")

# ---------- Actual vs Predicted ----------
st.subheader("🎯 Actual vs Predicted AQI")
sample_size = min(500, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)
av_df = pd.DataFrame({"Actual": y_test.values[idx], "Predicted": y_pred[idx]})

fig1 = px.scatter(av_df, x="Actual", y="Predicted", opacity=0.5,
                  color_discrete_sequence=["#3498db"],
                  title="Actual vs Predicted AQI")
fig1.add_shape(type="line",
               x0=av_df["Actual"].min(), y0=av_df["Actual"].min(),
               x1=av_df["Actual"].max(), y1=av_df["Actual"].max(),
               line=dict(color="red", dash="dash"))
fig1.update_layout(annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper",
                                     text="Red line = perfect prediction", showarrow=False)])
st.plotly_chart(fig1)

st.markdown("---")

# ---------- Residuals ----------
st.subheader("📉 Residual Analysis")
residuals = y_test.values - y_pred
res_df = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})

col1, col2 = st.columns(2)
with col1:
    fig2 = px.histogram(res_df, x="Residual", nbins=40,
                        color_discrete_sequence=["#e67e22"],
                        title="Residual Distribution")
    st.plotly_chart(fig2)
with col2:
    fig3 = px.scatter(res_df, x="Predicted", y="Residual", opacity=0.4,
                      color_discrete_sequence=["#9b59b6"],
                      title="Residuals vs Predicted")
    fig3.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig3)

st.caption(f"Model trained on data from: `{source}`")
