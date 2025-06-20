import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'location', 'new_cases_smoothed']].dropna()

df = load_data()

st.title("COVID-19 Trend Forecaster")
st.markdown("""
Forecast pandemic trends using ARIMA modeling with real-world data from 
[Our World in Data](https://github.com/owid/covid-19-data)
""")

st.sidebar.header("Configuration")
country = st.sidebar.selectbox("Select Country", df['location'].unique(), index=70)  # Default: India
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
train_size = st.sidebar.slider("Training Data (%)", 70, 95, 80)

country_data = df[df['location'] == country].set_index('date')
ts_data = country_data['new_cases_smoothed'].fillna(0)

split_idx = int(len(ts_data) * train_size / 100)
train, test = ts_data[:split_idx], ts_data[split_idx:]

model = ARIMA(train, order=(7,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_days)
forecast_dates = pd.date_range(train.index[-1], periods=forecast_days+1, freq='D')[1:]

test_forecast = model_fit.forecast(steps=len(test))
mae = mean_absolute_error(test, test_forecast[:len(test)])
rmse = np.sqrt(mean_squared_error(test, test_forecast[:len(test)]))

fig = px.line(title=f'COVID-19 Cases in {country}')
fig.add_scatter(x=train.index, y=train, name='Training Data', line=dict(color='blue'))
fig.add_scatter(x=test.index, y=test, name='Actual Trend', line=dict(color='green'))
fig.add_scatter(x=forecast_dates, y=forecast, name='Forecast', line=dict(color='red', dash='dash'))
fig.add_scatter(x=test.index, y=test_forecast[:len(test)], name='Test Forecast', line=dict(color='purple'))
fig.update_layout(xaxis_title='Date', yaxis_title='Smoothed Daily Cases', hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)
st.metric("Model Performance (Test Set)", f"MAE: {mae:.1f} | RMSE: {rmse:.1f}")

with st.expander("Data Summary"):
    st.dataframe(country_data.describe(), use_container_width=True)
    st.download_button("Download Processed Data", country_data.to_csv(), f"{country}_covid_data.csv")
