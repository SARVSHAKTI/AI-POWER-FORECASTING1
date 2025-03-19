import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/Cleaned_dataset.csv", parse_dates=["ds"])
    return df

df = load_data()

# Ensure the date column is in datetime format
df = df.rename(columns={"MCP (Rs/MWh)": "y"})  # Rename target column

df = df.sort_values(by="ds")  # Ensure the dataset is sorted by date

# Prophet Model
st.title("📈 Prophet Time Series Forecasting")

# Train Prophet Model
model = Prophet()
model.fit(df)

# Create future dates for forecasting
future = model.make_future_dataframe(periods=30, freq="D")  # Predict next 30 days
forecast = model.predict(future)

# Compute RMSE & MAPE
actuals = df.set_index("ds")["y"]
predicted = forecast.set_index("ds").loc[actuals.index, "yhat"]

rmse = np.sqrt(mean_squared_error(actuals, predicted))
mape = mean_absolute_percentage_error(actuals, predicted) * 100

st.write("### Model Performance")
st.write(f"✅ RMSE: {rmse:.2f}")
st.write(f"✅ MAPE: {mape:.2f}%")

# Plot Forecast
st.write("### Forecast Plot")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

# Download Forecast Data
csv = forecast.to_csv(index=False).encode("utf-8")
st.download_button("Download Forecast Data", csv, "prophet_forecast.csv", "text/csv")
