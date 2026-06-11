# 🌫️ Air Quality Index Prediction System

A machine learning web application that predicts the **Air Quality Index (AQI)** based on pollutant concentrations, built with Python, Scikit-learn, and Streamlit.

---

## 📁 Project Structure

```
aqi_project/
├── app.py                          # 🏠 Main prediction page
├── utils.py                        # 🔧 Shared helper functions
├── requirements.txt                # 📦 Dependencies
├── aqi_model.pkl                   # 🤖 Trained ML model
├── AQI_CLEANED.csv                 # 🗃️ Cleaned dataset
├── .streamlit/
│   └── config.toml                 # 🎨 App theme
└── pages/
    ├── 1_EDA.py                    # 📊 Exploratory Data Analysis
    ├── 2_Feature_Importance.py     # 🧠 Model explainability
    ├── 3_Model_Performance.py      # 📈 Metrics & evaluation
    └── 4_About.py                  # ℹ️ Project information
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure these files exist in the project folder:
- `aqi_model.pkl` — trained model (from your `model_building.ipynb`)
- `AQI_CLEANED.csv` — cleaned dataset (from your `datapreprocessing.ipynb`)

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source:** [CPCB India Air Quality Data](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **File:** `city_day.csv`
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target:** AQI

---

## 🏷️ AQI Scale

| AQI | Category | Impact |
|-----|----------|--------|
| 0–50 | 🟢 Good | Minimal |
| 51–100 | 🟡 Satisfactory | Minor breathing discomfort |
| 101–200 | 🟠 Moderate | Breathing discomfort for sensitive groups |
| 201–300 | 🔴 Poor | Discomfort for most |
| 301–400 | 🟣 Very Poor | Illness risk |
| 400+ | ⚫ Severe | Emergency |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML:** Scikit-learn
- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **Deployment:** Streamlit Cloud / Local
