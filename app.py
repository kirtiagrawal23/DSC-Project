
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 House Price Predictor")

uploaded_file = st.file_uploader("Upload train.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, drop_first=True)

    df['SalePrice'] = np.log(df['SalePrice'])

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Model Performance")
    st.write("R² Score:", r2_score(y_test, y_pred))

    st.subheader("SalePrice Distribution")
    fig, ax = plt.subplots()
    ax.hist(np.exp(df['SalePrice']), bins=50)
    st.pyplot(fig)

    st.subheader("Prediction (Sample)")

    sample = X_test.iloc[0].values.reshape(1, -1)
    pred_log = model.predict(sample)
    pred_price = np.exp(pred_log)

    st.write("Predicted Price:", pred_price[0])
    st.write("Actual Price:", np.exp(y_test.iloc[0]))
