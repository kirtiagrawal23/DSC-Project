import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ─── Page Config ─────────────────────────

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ─── Load Data ───────────────────────────

@st.cache_data
def load_data():
df = pd.read_csv("cleaned_data.csv")
pred = pd.read_csv("predictions.csv")
return df, pred

@st.cache_resource
def load_model():
return joblib.load("model.pkl")

@st.cache_data
def load_json(file):
with open(file) as f:
return json.load(f)

# Load all files

df, pred_df = load_data()
model = load_model()
feature_names = load_json("feature_names.json")
metrics = load_json("model_metrics.json")
feat_imp = load_json("feature_importance.json")

# ─── Tabs ────────────────────────────────

tab1, tab2, tab3 = st.tabs([
"📊 Dashboard",
"🔮 Prediction Engine",
"📈 Model Accuracy"
])

# ════════════════════════════════════════

# 📊 TAB 1 — DASHBOARD

# ════════════════════════════════════════

with tab1:
st.title("📊 Data Dashboard")

```
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sale Price Distribution")
    st.bar_chart(df["SalePrice"])

with col2:
    st.subheader("Top Correlated Features")
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)[1:10]
    st.bar_chart(corr)
```

# ════════════════════════════════════════

# 🔮 TAB 2 — PREDICTION ENGINE

# ════════════════════════════════════════

with tab2:
st.title("🔮 Predict House Price")

```
st.write("Enter some feature values:")

user_input = {}

# Take limited inputs for simplicity
for col in feature_names[:10]:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict Price"):
    input_df = pd.DataFrame([user_input])

    # Ensure all columns match training
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    pred_log = model.predict(input_df)[0]
    pred_price = np.exp(pred_log)

    st.success(f"💰 Predicted House Price: ${pred_price:,.0f}")
```

# ════════════════════════════════════════

# 📈 TAB 3 — MODEL ACCURACY

# ════════════════════════════════════════

with tab3:
st.title("📈 Model Performance")

```
m = metrics["Linear Regression"]

col1, col2, col3 = st.columns(3)
col1.metric("R² Score", round(m["R2"], 4))
col2.metric("MAE", round(m["MAE"], 4))
col3.metric("RMSE", round(m["RMSE"], 4))

st.subheader("Actual vs Predicted")

chart_df = pred_df.rename(columns={
    "actual": "Actual Price",
    "predicted": "Predicted Price"
})

st.scatter_chart(chart_df)
```

# ─── Footer ─────────────────────────────

st.markdown("---")
st.markdown("Built with Streamlit · House Price Prediction Project")
