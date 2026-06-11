import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    if not os.path.exists("aqi_model.pkl"):
        st.error("❌ Model file 'aqi_model.pkl' not found. Please train the model first.")
        st.stop()
    return joblib.load("aqi_model.pkl")

model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_the_Clouds.jpg/320px-Above_the_Clouds.jpg", use_container_width=True)
    st.title("ℹ️ About")
    st.markdown("""
    This app predicts the **Air Quality Index (AQI)** based on pollutant concentrations.

    **AQI Categories:**
    | Range | Category |
    |-------|----------|
    | 0–50 | 🟢 Good |
    | 51–100 | 🟡 Satisfactory |
    | 101–200 | 🟠 Moderate |
    | 201–300 | 🔴 Poor |
    | 301–400 | 🟣 Very Poor |
    | 400+ | ⚫ Severe |

    **Data Source:** CPCB India (city_day.csv)
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn")

# ---------- Header ----------
st.title("🌫️ Air Quality Index Prediction System")
st.markdown("Enter pollutant concentrations below to predict the AQI for a location.")
st.markdown("---")

# ---------- Input Form ----------
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

# ---------- Predict ----------
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    predict_clicked = st.button("🔍 Predict AQI", use_container_width=True)
with col_btn2:
    if st.button("🔄 Reset", use_container_width=False):
        st.rerun()

if predict_clicked:
    input_data = pd.DataFrame({
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

    prediction = model.predict(input_data)[0]

    # AQI Category & Color
    if prediction <= 50:
        category, color, emoji = "Good",      "#2ecc71", "🟢"
    elif prediction <= 100:
        category, color, emoji = "Satisfactory", "#f1c40f", "🟡"
    elif prediction <= 200:
        category, color, emoji = "Moderate",  "#e67e22", "🟠"
    elif prediction <= 300:
        category, color, emoji = "Poor",      "#e74c3c", "🔴"
    elif prediction <= 400:
        category, color, emoji = "Very Poor", "#8e44ad", "🟣"
    else:
        category, color, emoji = "Severe",    "#2c3e50", "⚫"

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown(f"""
        <div style="
            background-color: {color}22;
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 24px;
            text-align: center;
        ">
            <h1 style="color:{color}; font-size:64px; margin:0;">{prediction:.1f}</h1>
            <p style="font-size:18px; margin:4px 0;">Predicted AQI</p>
            <h3 style="color:{color};">{emoji} {category}</h3>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        # Health advice per category
        advice = {
            "Good":        "Air quality is excellent. No precautions needed. ✅",
            "Satisfactory":"Air quality is acceptable. Sensitive individuals should limit prolonged outdoor activity.",
            "Moderate":    "Members of sensitive groups may experience health effects. Wear a mask outdoors. 😷",
            "Poor":        "Everyone may begin to experience health effects. Avoid outdoor activity if possible. ⚠️",
            "Very Poor":   "Health alert: Serious health effects for everyone. Stay indoors. 🚨",
            "Severe":      "Emergency conditions. Entire population is likely to be affected. Do NOT go outside. ☠️",
        }
        st.info(f"**Health Advisory:** {advice[category]}")
        st.markdown("**Your Input Summary:**")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
