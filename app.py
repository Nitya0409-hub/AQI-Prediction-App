import streamlit as st
import pandas as pd
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("aqi_model.pkl"):
        st.error("❌ Model file 'aqi_model.pkl' not found. Please train the model first.")
        st.stop()
    return joblib.load("aqi_model.pkl")

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ AQI Prediction")
    st.markdown("---")
    st.markdown("""
    ### AQI Categories
    | Range | Category |
    |-------|----------|
    | 0–50 | 🟢 Good |
    | 51–100 | 🟡 Satisfactory |
    | 101–200 | 🟠 Moderate |
    | 201–300 | 🔴 Poor |
    | 301–400 | 🟣 Very Poor |
    | 400+ | ⚫ Severe |
    """)
    st.markdown("---")
    st.caption("Dataset: AQI_SMALL_5000.csv (CPCB India)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌫️ Air Quality Index Prediction System")
st.markdown("Enter pollutant concentrations below to predict the AQI.")
st.markdown("---")

# ── Input Form (3 columns) ────────────────────────────────────────────────────
st.subheader("📥 Enter Pollutant Values")

col1, col2, col3 = st.columns(3)

with col1:
    pm25    = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    pm10    = st.number_input("PM10 (µg/m³)",   min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    no      = st.number_input("NO (µg/m³)",     min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    no2     = st.number_input("NO2 (µg/m³)",    min_value=0.0, max_value=999.0, value=0.0, step=0.1)

with col2:
    nox     = st.number_input("NOx (ppb)",      min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    nh3     = st.number_input("NH3 (µg/m³)",    min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    co      = st.number_input("CO (mg/m³)",     min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    so2     = st.number_input("SO2 (µg/m³)",    min_value=0.0, max_value=999.0, value=0.0, step=0.1)

with col3:
    o3      = st.number_input("O3 (µg/m³)",     min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    benzene = st.number_input("Benzene (µg/m³)",min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    toluene = st.number_input("Toluene (µg/m³)",min_value=0.0, max_value=999.0, value=0.0, step=0.1)
    xylene  = st.number_input("Xylene (µg/m³)", min_value=0.0, max_value=999.0, value=0.0, step=0.1)

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict AQI", use_container_width=False)

if predict_clicked:
    data = pd.DataFrame({
        'PM2.5':   [pm25],
        'PM10':    [pm10],
        'NO':      [no],
        'NO2':     [no2],
        'NOx':     [nox],
        'NH3':     [nh3],
        'CO':      [co],
        'SO2':     [so2],
        'O3':      [o3],
        'Benzene': [benzene],
        'Toluene': [toluene],
        'Xylene':  [xylene]
    })

    prediction = model.predict(data)[0]

    # Category + color + advice
    if prediction <= 50:
        category, color, emoji = "Good",         "#2ecc71", "🟢"
        advice = "Air quality is excellent. No health precautions needed."
    elif prediction <= 100:
        category, color, emoji = "Satisfactory", "#f1c40f", "🟡"
        advice = "Acceptable air quality. Sensitive individuals should limit prolonged outdoor activity."
    elif prediction <= 200:
        category, color, emoji = "Moderate",     "#e67e22", "🟠"
        advice = "Breathing discomfort for asthma patients. Consider wearing a mask outdoors."
    elif prediction <= 300:
        category, color, emoji = "Poor",         "#e74c3c", "🔴"
        advice = "Breathing discomfort for most people. Avoid outdoor activity if possible."
    elif prediction <= 400:
        category, color, emoji = "Very Poor",    "#8e44ad", "🟣"
        advice = "Risk of respiratory illness on prolonged exposure. Stay indoors."
    else:
        category, color, emoji = "Severe",       "#2c3e50", "⚫"
        advice = "Emergency conditions. Do NOT go outside."

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown(f"""
        <div style="
            background-color:{color}22;
            border-left: 6px solid {color};
            border-radius: 10px;
            padding: 28px;
            text-align: center;
        ">
            <h1 style="color:{color}; font-size:72px; margin:0;">{prediction:.1f}</h1>
            <p style="font-size:18px; margin:4px 0; color:gray;">Predicted AQI</p>
            <h3 style="color:{color}; margin-top:8px;">{emoji} {category}</h3>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.info(f"**🏥 Health Advisory:** {advice}")
        st.markdown("**Your Input Summary:**")
        st.dataframe(data.T.rename(columns={0: "Value"}), use_container_width=True)
