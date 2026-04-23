import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="House Price Predictor", layout="wide")

# ─── Load Files ─────────────────────────

@st.cache_data
def load_data():
df = pd.read_csv("cleaned_data.csv")
pred = pd.read_csv("predictions.csv")
return df, pred

@st.cache_resource
def load_model():
return joblib.load("model.pkl")

df, pred_df = load_data()
model = load_model()

with open("feature_names.json") as f:
feature_names = json.load(f)

with open("model_metrics.json") as f:
metrics = json.load(f)

with open("feature_importance.json") as f:
feat_imp = json.load(f)

# ─── Tabs ───────────────────────────────

tab1, tab2, tab3 = st.tabs([
"📊 Dashboard",
"🔮 Prediction Engine",
"📈 Model Accuracy"
])

# ═══════════════════════════════════════

# 📊 TAB 1 — DASHBOARD

# ═══════════════════════════════════════

with tab1:
st.title("📊 Data Dashboard")

```
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sale Price Distribution")
    st.bar_chart(df["SalePrice"])

with col2:
    st.subheader("Top Correlations")
    corr = df.corr()["SalePrice"].sort_values(ascending=False)[1:10]
    st.bar_chart(corr)
```

# ═══════════════════════════════════════

# 🔮 TAB 2 — PREDICTION

# ═══════════════════════════════════════

with tab2:
st.title("🔮 Predict House Price")

```
user_input = {}

for col in feature_names[:10]:   # limiting inputs
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    pred_log = model.predict(input_df)[0]
    pred_price = np.exp(pred_log)

    st.success(f"💰 Predicted Price: ${pred_price:,.0f}")
```

# ═══════════════════════════════════════

# 📈 TAB 3 — MODEL ACCURACY

# ═══════════════════════════════════════

with tab3:
st.title("📈 Model Performance")

```
m = metrics["Linear Regression"]

st.metric("R2 Score", round(m["R2"], 4))
st.metric("MAE", round(m["MAE"], 4))
st.metric("RMSE", round(m["RMSE"], 4))

st.subheader("Actual vs Predicted")

st.scatter_chart(pred_df.rename(columns={
    "actual": "Actual",
    "predicted": "Predicted"
}))
```
