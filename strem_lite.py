import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:/360DigitMG Project/Project 2 Power forecasting/Dataset/Cleaned_dataset.csv")
    return df

df = load_data()

# Ensure the date column is in datetime format
df["ds"] = pd.date_range(start="2023-01-01", periods=len(df))

# Select target column
target_column = "MCP (Rs/MWh)"  # Change this to your actual target column name
df.rename(columns={target_column: "y"}, inplace=True)

# Prophet Model
st.title("📈 Prophet Time Series Forecasting")

# Train Prophet Model
model = Prophet()
model.fit(df)

# Create future dates for forecasting
future = model.make_future_dataframe(periods=30)  # Predict next 30 days
forecast = model.predict(future)

# Compute RMSE & MAPE
rmse = np.sqrt(mean_squared_error(df["y"], forecast["yhat"][:len(df)]))
mape = mean_absolute_percentage_error(df["y"], forecast["yhat"][:len(df)]) * 100

st.write("### Model Performance")
st.write(f"✅ RMSE: {rmse:.2f}")
st.write(f"✅ MAPE: {mape:.2f}%")

# Plot Forecast
st.write("### Forecast Plot")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

# Download Forecast Data
csv = forecast.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast Data", csv, "prophet_forecast.csv", "text/csv")