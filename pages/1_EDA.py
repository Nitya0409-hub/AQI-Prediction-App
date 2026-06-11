import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EDA - AQI", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")
st.markdown("Insights from the CPCB India Air Quality dataset.")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    for path in ["AQI_CLEANED.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path), path
    return None, None

df, source = load_data()

if df is None:
    st.warning("⚠️ No dataset found. Please place `AQI_CLEANED.csv` or `city_day.csv` in the project folder.")
    st.stop()

st.caption(f"Loaded from: `{source}` — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------- Overview ----------
st.subheader("🔎 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Features", df.shape[1])
if "AQI" in df.columns:
    col3.metric("Avg AQI", f"{df['AQI'].mean():.1f}")
    col4.metric("Max AQI", f"{df['AQI'].max():.0f}")

with st.expander("📋 Show Raw Data (first 100 rows)"):
    st.dataframe(df.head(100))

with st.expander("📈 Descriptive Statistics"):
    st.dataframe(df.describe())

st.markdown("---")

# ---------- AQI Distribution ----------
if "AQI" in df.columns:
    st.subheader("📉 AQI Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="AQI", nbins=50, color_discrete_sequence=["#3498db"],
                           title="AQI Frequency Distribution")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig)

    with col2:
        fig2 = px.box(df, y="AQI", color_discrete_sequence=["#e74c3c"],
                      title="AQI Boxplot (Outlier View)")
        st.plotly_chart(fig2)

    # AQI Category breakdown
    def categorize(aqi):
        if aqi <= 50:   return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else:            return "Severe"

    df["AQI_Category"] = df["AQI"].apply(categorize)
    cat_counts = df["AQI_Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]

    color_map = {
        "Good": "#2ecc71", "Satisfactory": "#f1c40f", "Moderate": "#e67e22",
        "Poor": "#e74c3c", "Very Poor": "#8e44ad", "Severe": "#2c3e50"
    }

    fig3 = px.bar(cat_counts, x="Category", y="Count",
                  color="Category", color_discrete_map=color_map,
                  title="AQI Category Distribution")
    st.plotly_chart(fig3)

st.markdown("---")

# ---------- Pollutant Correlations ----------
pollutants = [c for c in ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene","AQI"]
              if c in df.columns]

if len(pollutants) > 2:
    st.subheader("🔗 Pollutant Correlation Heatmap")
    corr = df[pollutants].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     title="Correlation Matrix", aspect="auto")
    st.plotly_chart(fig4)

st.markdown("---")

# ---------- City-wise AQI ----------
if "City" in df.columns and "AQI" in df.columns:
    st.subheader("🏙️ City-wise Average AQI")
    city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
    city_aqi.columns = ["City", "Avg AQI"]

    top_n = st.slider("Show top N cities", 5, min(50, len(city_aqi)), 20)
    fig5 = px.bar(city_aqi.head(top_n), x="City", y="Avg AQI",
                  color="Avg AQI", color_continuous_scale="Reds",
                  title=f"Top {top_n} Cities by Average AQI")
    fig5.update_xaxes(tickangle=45)
    st.plotly_chart(fig5)

st.markdown("---")

# ---------- Trend over Time ----------
if "Date" in df.columns and "AQI" in df.columns:
    st.subheader("📅 AQI Trend Over Time")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df_time = df.dropna(subset=["Date"]).sort_values("Date")
    monthly = df_time.resample("ME", on="Date")["AQI"].mean().reset_index()
    fig6 = px.line(monthly, x="Date", y="AQI", title="Monthly Average AQI Trend",
                   color_discrete_sequence=["#9b59b6"])
    st.plotly_chart(fig6)

# ---------- Pollutant vs AQI scatter ----------
if "AQI" in df.columns:
    st.subheader("🔬 Pollutant vs AQI Scatter")
    avail = [c for c in ["PM2.5","PM10","NO2","SO2","CO","O3"] if c in df.columns]
    if avail:
        chosen = st.selectbox("Select Pollutant", avail)
        sample = df[[chosen, "AQI"]].dropna().sample(min(2000, len(df)))
        fig7 = px.scatter(sample, x=chosen, y="AQI", opacity=0.4,
                          color_discrete_sequence=["#1abc9c"],
                          title=f"{chosen} vs AQI")
        st.plotly_chart(fig7)
