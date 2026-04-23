import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Page config

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load data

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

# Tabs

tab1, tab2, tab3 = st.tabs([
"Dashboard",
"Prediction Engine",
"Model Accuracy"
])

# ---------------- Dashboard ----------------

with tab1:
st.title("Dashboard")

```
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sale Price Distribution")
    st.bar_chart(df["SalePrice"])

with col2:
    st.subheader("Top Correlations")
    corr = df.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)[1:10]
    st.bar_chart(corr)
```

# ---------------- Prediction ----------------

with tab2:
st.title("Prediction Engine")

```
user_input = {}

for col in feature_names[:10]:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    pred_log = model.predict(input_df)[0]
    pred_price = np.exp(pred_log)

    st.success(f"Predicted Price: ${pred_price:,.0f}")
```

# ---------------- Model Accuracy ----------------

with tab3:
st.title("Model Accuracy")

```
m = metrics["Linear Regression"]

st.metric("R2 Score", round(m["R2"], 4))
st.metric("MAE", round(m["MAE"], 4))
st.metric("RMSE", round(m["RMSE"], 4))

st.subheader("Actual vs Predicted")

chart_df = pred_df.rename(columns={
    "actual": "Actual",
    "predicted": "Predicted"
})

st.scatter_chart(chart_df)
```
