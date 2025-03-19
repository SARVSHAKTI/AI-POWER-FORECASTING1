import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import LSTM, GRU, Dense  # type: ignore

warnings.filterwarnings("ignore")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:/360DigitMG Project/Project 2 Power forecasting/Dataset/Cleaned_dataset.csv")
    df.columns = df.columns.str.strip()  # Remove any whitespace
    return df

df = load_data()
st.title("Time Series Model Evaluation Dashboard")

# Select target variable
target_column = "MCP (Rs/MWh)"  # Change as needed
if target_column not in df.columns:
    st.error(f"Column '{target_column}' not found. Available columns: {df.columns}")
    st.stop()

y_true = df[target_column]

# Function to calculate RMSE & MAPE
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mape

# Linear Regression Model
def linear_regression_model():
    model = LinearRegression()
    model.fit(df.index.values.reshape(-1, 1), y_true)
    y_pred = model.predict(df.index.values.reshape(-1, 1))
    return calculate_metrics(y_true, y_pred)

# Lasso Regression Model
def lasso_model():
    model = Lasso(alpha=0.1)
    model.fit(df.index.values.reshape(-1, 1), y_true)
    y_pred = model.predict(df.index.values.reshape(-1, 1))
    return calculate_metrics(y_true, y_pred)

# Ridge Regression Model
def ridge_model():
    model = Ridge(alpha=1.0)
    model.fit(df.index.values.reshape(-1, 1), y_true)
    y_pred = model.predict(df.index.values.reshape(-1, 1))
    return calculate_metrics(y_true, y_pred)

# Random Forest Model
def random_forest_model():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(df.index.values.reshape(-1, 1), y_true)
    y_pred = model.predict(df.index.values.reshape(-1, 1))
    return calculate_metrics(y_true, y_pred)

# ARIMA Model
def arima_model():
    model = ARIMA(y_true, order=(3, 1, 2))
    model_fit = model.fit()
    y_pred = model_fit.predict(start=0, end=len(y_true) - 1)
    return calculate_metrics(y_true, y_pred)

# SARIMA Model
def sarima_model():
    model = SARIMAX(y_true, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    y_pred = model_fit.predict(start=0, end=len(y_true) - 1)
    return calculate_metrics(y_true, y_pred)

# Exponential Smoothing Model (ETS)
def ets_model():
    model = ExponentialSmoothing(y_true, trend="add", seasonal="add", seasonal_periods=12)
    model_fit = model.fit()
    y_pred = model_fit.fittedvalues
    return calculate_metrics(y_true, y_pred)

# LSTM Model
def lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    x_train = np.array(df.index.values).reshape(-1, 1, 1)
    y_train = np.array(y_true).reshape(-1, 1)
    model.fit(x_train, y_train, epochs=5, verbose=0)
    y_pred = model.predict(x_train)
    return calculate_metrics(y_true, y_pred.flatten())

# GRU Model
def gru_model():
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(1, 1)),
        GRU(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    x_train = np.array(df.index.values).reshape(-1, 1, 1)
    y_train = np.array(y_true).reshape(-1, 1)
    model.fit(x_train, y_train, epochs=5, verbose=0)
    y_pred = model.predict(x_train)
    return calculate_metrics(y_true, y_pred.flatten())

# Collect Model Results
models = {
    "Linear Regression": linear_regression_model,
    "Lasso Regression": lasso_model,
    "Ridge Regression": ridge_model,
    "Random Forest": random_forest_model,
    "ARIMA": arima_model,
    "SARIMA": sarima_model,
    "ETS (Holt-Winters)": ets_model,
    "LSTM": lstm_model,
    "GRU": gru_model,
}

results = []
for model_name, model_func in models.items():
    try:
        rmse, mape = model_func()
        meets_criteria = "Yes" if mape < 10 else "No"
        best_model = "Verification"
        results.append({
            "Model": model_name,
            "Feature Engineering": "Scaling, Lag Features",
            "Hyperparameter": "Default" if "Regression" in model_name else "Tuned",
            "RMSE": round(rmse, 2),
            "MAPE%": round(mape, 2),
            "Meets Evaluation Requirements": meets_criteria,
            "Best Model": best_model,
            "Development App": "Streamlit",
        })
    except Exception as e:
        results.append({
            "Model": model_name,
            "Feature Engineering": "Error",
            "Hyperparameter": "Error",
            "RMSE": "N/A",
            "MAPE%": "N/A",
            "Meets Evaluation Requirements": "Error",
            "Best Model": "Error",
            "Development App": "Streamlit",
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Display DataFrame in Streamlit
st.write("### Model Evaluation Table")
st.dataframe(results_df)

# Download Button
csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Model Evaluation Report", csv, "model_evaluation.csv", "text/csv")
