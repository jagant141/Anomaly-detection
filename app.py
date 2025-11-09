import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import io, zipfile, requests

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Anomalous User Behavior Detection", layout="wide")

# ========== APP TITLE ==========
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🔍 Anomalous User Behavior Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# =========================================
# FUNCTION TO LOAD DATA FROM URL OR FILE
# =========================================
def load_dataset(file_or_url, is_url=True):
    """Load dataset from URL or uploaded file. Supports CSV, Excel, JSON, Parquet, TXT, ZIP."""
    try:
        if is_url:
            file_ext = file_or_url.split('.')[-1].lower()
            response = requests.get(file_or_url)
            response.raise_for_status()
            content = io.BytesIO(response.content)
        else:
            file_ext = file_or_url.name.split('.')[-1].lower()
            content = file_or_url

        if file_ext == "csv":
            df = pd.read_csv(content)
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(content)
        elif file_ext == "json":
            df = pd.read_json(content)
        elif file_ext == "parquet":
            df = pd.read_parquet(content)
        elif file_ext in ["txt", "log"]:
            df = pd.read_csv(content, delimiter=None, engine="python")
        elif file_ext == "zip":
            with zipfile.ZipFile(content) as z:
                file_name = z.namelist()[0]
                with z.open(file_name) as f:
                    try:
                        df = pd.read_csv(f)
                    except:
                        df = pd.read_excel(f)
        else:
            st.error("Unsupported file type. Please upload CSV, Excel, JSON, Parquet, TXT, or ZIP.")
            return None
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

# =========================================
# MAIN UI SECTION
# =========================================
st.sidebar.title("📂 Data Input")
source_type = st.sidebar.selectbox(
    "Choose how to load your dataset:",
    ["-- Select Option --", "📁 Upload File", "🌐 Provide URL"]
)

df = None

# Upload option
if source_type == "📁 Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset file",
        type=["csv", "xlsx", "json", "parquet", "zip", "txt"]
    )
    if uploaded_file is not None:
        df = load_dataset(uploaded_file, is_url=False)

# URL option
elif source_type == "🌐 Provide URL":
    dataset_url = st.sidebar.text_input("Enter dataset URL:")
    if dataset_url:
        df = load_dataset(dataset_url, is_url=True)

# =========================================
# WHEN DATA IS LOADED
# =========================================
if df is not None:
    st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # --- Data Preprocessing ---
    df_processed = df.copy()

    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            try:
                df_processed[col] = pd.to_datetime(df[col], errors='coerce')
                df_processed[f"{col}_hour"] = df_processed[col].dt.hour
                df_processed[f"{col}_day"] = df_processed[col].dt.day
                df_processed[f"{col}_weekday"] = df_processed[col].dt.weekday
            except Exception:
                pass

    # Encode categorical features
    df_encoded = pd.get_dummies(df_processed, drop_first=True)
    df_encoded.replace([np.inf, -np.inf], 0, inplace=True)
    df_encoded.fillna(0, inplace=True)

    # Run button
    run_analysis = st.button("🚀 Run Anomaly Detection")

    if run_analysis:
        try:
            # Train Isolation Forest
            scaler = StandardScaler()
            X = scaler.fit_transform(df_encoded.select_dtypes(include=[np.number]))
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X)

            # Predict anomalies
            df["anomaly_score"] = model.decision_function(X)
            df["is_anomaly"] = model.predict(X)

            anomalies = df[df["is_anomaly"] == -1]
            st.success(f"✅ Detected {len(anomalies)} anomalies out of {len(df)} total records.")
            st.write("### 🧾 Detected Anomalies:")
            st.dataframe(anomalies.head(20))

            # ---------- DOWNLOAD BUTTON ----------
            csv = anomalies.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Detected Anomalies (CSV)",
                data=csv,
                file_name="detected_anomalies.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error processing or analyzing dataset: {e}")
