import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Air Quality Monitor & Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class AirQualityPredictor:
    def __init__(self):
        self.api_key = None
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        
    def get_coordinates(self, city_name):
        """Get coordinates for a city using OpenWeatherMap Geocoding API"""
        if not self.api_key:
            return None, None
            
        try:
            params = {
                'q': city_name,
                'limit': 1,
                'appid': self.api_key
            }
            response = requests.get(self.geocoding_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data[0]['lat'], data[0]['lon']
        except Exception as e:
            st.error(f"Error getting coordinates: {str(e)}")
        return None, None
    
    def get_current_air_quality(self, lat, lon):
        """Fetch current air quality data"""
        if not self.api_key:
            return None
            
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching air quality data: {str(e)}")
        return None
    
    def get_historical_data(self, lat, lon, start_time, end_time):
        """Fetch historical air quality data"""
        if not self.api_key:
            return None
            
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'start': start_time,
                'end': end_time,
                'appid': self.api_key
            }
            response = requests.get(f"{self.base_url}/history", params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
        return None
    
    def generate_synthetic_data(self, days=30):
        """Generate realistic synthetic air quality data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=days), 
                             end=datetime.datetime.now(), freq='H')
        
        base_aqi = 50
        seasonal_pattern = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 7))
        daily_pattern = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) 
        noise = np.random.normal(0, 8, len(dates))
        
        aqi_values = base_aqi + seasonal_pattern + daily_pattern + noise
        aqi_values = np.clip(aqi_values, 0, 300) 
        
        pm25 = aqi_values * 0.4 + np.random.normal(0, 3, len(dates))
        pm10 = aqi_values * 0.6 + np.random.normal(0, 5, len(dates))
        co = aqi_values * 0.02 + np.random.normal(0, 0.1, len(dates))
        no2 = aqi_values * 0.3 + np.random.normal(0, 2, len(dates))
        o3 = aqi_values * 0.5 + np.random.normal(0, 4, len(dates))
        
        pm25 = np.maximum(pm25, 0)
        pm10 = np.maximum(pm10, 0)
        co = np.maximum(co, 0)
        no2 = np.maximum(no2, 0)
        o3 = np.maximum(o3, 0)
        
        df = pd.DataFrame({
            'datetime': dates,
            'aqi': aqi_values,
            'pm2_5': pm25,
            'pm10': pm10,
            'co': co,
            'no2': no2,
            'o3': o3,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        df = df.copy()

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        for col in ['pm2_5', 'pm10', 'co', 'no2', 'o3']:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag24'] = df[col].shift(24)
        
        df['aqi_rolling_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
        df['pm2_5_rolling_24h'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
        
        return df.dropna()
    
    def train_models(self, df):
        """Train multiple models for AQI prediction"""
        df_features = self.prepare_features(df)
        
        feature_cols = [col for col in df_features.columns if col not in ['datetime', 'aqi', 'hour', 'day_of_week', 'month']]
        X = df_features[feature_cols]
        y = df_features['aqi']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'R²': r2,
                'predictions': y_pred,
                'actual': y_test.values
            }
            
            trained_models[name] = model
        
        return results, trained_models, scaler, feature_cols
    
    def predict_future(self, model, scaler, df, feature_cols, hours_ahead=24):
        """Predict future AQI values"""
        df_features = self.prepare_features(df)
        
        last_row = df_features.iloc[-1:][feature_cols]
        
        predictions = []
        current_data = df_features.copy()
        
        for i in range(hours_ahead):

            if isinstance(model, LinearRegression):
                pred_scaled = scaler.transform(last_row)
                pred = model.predict(pred_scaled)[0]
            else:
                pred = model.predict(last_row)[0]
            
            predictions.append(pred)
            
        return predictions

def get_aqi_category(aqi_value):
    """Get AQI category and color"""
    if aqi_value <= 50:
        return "Good", "#00E400"
    elif aqi_value <= 100:
        return "Moderate", "#FFFF00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi_value <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

def main():
    st.markdown('<h1 class="main-header">🌍 Air Quality Monitor & Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**Monitor real-time air quality and predict future trends using machine learning**")
    
    predictor = AirQualityPredictor()
    

    st.sidebar.header("🔧 Configuration")
    
    api_key = st.sidebar.text_input("OpenWeatherMap API Key (Optional)", type="password", 
                                   help="Enter your API key to fetch real data, or use demo mode with synthetic data")
    
    if api_key:
        predictor.api_key = api_key
    
    default_cities = ["New York", "London", "Tokyo", "Delhi", "Beijing", "Los Angeles", "Mumbai"]
    selected_city = st.sidebar.selectbox("Select City", default_cities + ["Custom"])
    
    if selected_city == "Custom":
        custom_city = st.sidebar.text_input("Enter City Name")
        if custom_city:
            selected_city = custom_city
    
    # Data source
    use_demo_data = st.sidebar.checkbox("Use Demo Data", value=not bool(api_key),
                                       help="Use synthetic data for demonstration")
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    prediction_hours = st.sidebar.slider("Hours to Predict", 1, 72, 24)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Air Quality Analysis")
        
        if use_demo_data or not api_key:
            st.info("Using synthetic data for demonstration. Enter an API key to use real data.")
            df = predictor.generate_synthetic_data(days=30)
            current_aqi = df['aqi'].iloc[-1]
            current_pm25 = df['pm2_5'].iloc[-1]
        else:
            lat, lon = predictor.get_coordinates(selected_city)
            if lat and lon:
                current_data = predictor.get_current_air_quality(lat, lon)
                if current_data:
                    aqi_data = current_data['list'][0]
                    current_aqi = aqi_data['main']['aqi'] * 50 
                    current_pm25 = aqi_data['components'].get('pm2_5', 0)
                    
                    df = predictor.generate_synthetic_data(days=30)
                    df['aqi'].iloc[-1] = current_aqi
                    df['pm2_5'].iloc[-1] = current_pm25
                else:
                    st.error("Could not fetch real data. Using demo data instead.")
                    df = predictor.generate_synthetic_data(days=30)
                    current_aqi = df['aqi'].iloc[-1]
                    current_pm25 = df['pm2_5'].iloc[-1]
            else:
                st.error("Could not find city coordinates. Using demo data instead.")
                df = predictor.generate_synthetic_data(days=30)
                current_aqi = df['aqi'].iloc[-1]
                current_pm25 = df['pm2_5'].iloc[-1]
        
        category, color = get_aqi_category(current_aqi)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Air Quality Index Over Time', 'Pollutant Concentrations'),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['aqi'], name='AQI', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['pm2_5'], name='PM2.5', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['pm10'], name='PM10', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['no2'], name='NO2', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text=f"Air Quality Data for {selected_city}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("🎯 Current Status")
        
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: {color};">Current AQI: {current_aqi:.0f}</h3>
            <h4 style="color: {color};">{category}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Pollutant Levels")
        st.metric("PM2.5", f"{current_pm25:.1f} μg/m³")
        st.metric("PM10", f"{df['pm10'].iloc[-1]:.1f} μg/m³")
        st.metric("NO2", f"{df['no2'].iloc[-1]:.1f} μg/m³")
        st.metric("CO", f"{df['co'].iloc[-1]:.2f} mg/m³")
        st.metric("O3", f"{df['o3'].iloc[-1]:.1f} μg/m³")
    st.header("🤖 Machine Learning Predictions")
    
    with st.spinner("Training models and generating predictions..."):
        results, trained_models, scaler, feature_cols = predictor.train_models(df)

        st.subheader("Model Performance")
        
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'MAE': [results[model]['MAE'] for model in results.keys()],
            'MSE': [results[model]['MSE'] for model in results.keys()],
            'R²': [results[model]['R²'] for model in results.keys()]
        })
        
        st.dataframe(performance_df, use_container_width=True)

        best_model_name = min(results.keys(), key=lambda x: results[x]['MAE'])
        st.success(f"Best performing model: **{best_model_name}** (Lowest MAE: {results[best_model_name]['MAE']:.2f})")
        
        best_model = trained_models[best_model_name]
        future_predictions = predictor.predict_future(best_model, scaler, df, feature_cols, prediction_hours)
        
        current_time = df['datetime'].iloc[-1]
        future_times = pd.date_range(start=current_time + pd.Timedelta(hours=1), 
                                   periods=prediction_hours, freq='H')
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=df['datetime'].tail(72), 
            y=df['aqi'].tail(72),
            name='Historical AQI',
            line=dict(color='blue')
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=future_times,
            y=future_predictions,
            name='Predicted AQI',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title=f"AQI Prediction for Next {prediction_hours} Hours",
            xaxis_title="Time",
            yaxis_title="Air Quality Index",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)

        avg_predicted_aqi = np.mean(future_predictions)
        pred_category, pred_color = get_aqi_category(avg_predicted_aqi)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Predicted AQI", f"{avg_predicted_aqi:.0f}")
        with col2:
            st.markdown(f"**Predicted Category:** <span style='color: {pred_color};'>{pred_category}</span>", 
                       unsafe_allow_html=True)
        with col3:
            trend = "Improving" if avg_predicted_aqi < current_aqi else "Worsening"
            trend_color = "green" if trend == "Improving" else "red"
            st.markdown(f"**Trend:** <span style='color: {trend_color};'>{trend}</span>", 
                       unsafe_allow_html=True)
    
    st.header("💡 Health Recommendations")
    
    if avg_predicted_aqi <= 50:
        st.success("Air quality is good. Great time for outdoor activities!")
    elif avg_predicted_aqi <= 100:
        st.warning("Air quality is moderate. Consider reducing prolonged outdoor exertion.")
    elif avg_predicted_aqi <= 150:
        st.warning("Unhealthy for sensitive groups. People with respiratory conditions should limit outdoor activities.")
    elif avg_predicted_aqi <= 200:
        st.error("Air quality is unhealthy. Everyone should reduce outdoor activities.")
    else:
        st.error("Air quality is very unhealthy or hazardous. Avoid outdoor activities and use air purifiers indoors.")
    
    st.header("📥 Download Data")
    
    download_df = df.copy()
    download_df['predicted_aqi'] = np.nan
    
    pred_df = pd.DataFrame({
        'datetime': future_times,
        'aqi': np.nan,
        'pm2_5': np.nan,
        'pm10': np.nan,
        'co': np.nan,
        'no2': np.nan,
        'o3': np.nan,
        'hour': future_times.hour,
        'day_of_week': future_times.dayofweek,
        'month': future_times.month,
        'predicted_aqi': future_predictions
    })
    
    combined_df = pd.concat([download_df, pred_df], ignore_index=True)
    
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download Complete Dataset (CSV)",
        data=csv,
        file_name=f"air_quality_data_{selected_city}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("""
    **About this App:**
    This application demonstrates machine learning techniques for environmental monitoring and prediction. 
    It uses real-world air quality data when available, or synthetic data that mimics realistic patterns 
    for demonstration purposes. The models trained include Random Forest and Linear Regression algorithms 
    to predict future air quality trends.
    
    **Data Sources:** OpenWeatherMap API (when API key provided), Synthetic data for demonstration
    """)

if __name__ == "__main__":
    main()
