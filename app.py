import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from weather_monitor import get_current_temp, get_season_stats, check_anomaly, get_current_season
from analysis import calc_moving_avg, calc_anomalies, calc_trend, calc_city_season_stats

st.set_page_config(page_title="Temperature Analysis", layout="wide")
st.title("Temperature Analysis and Monitoring")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload temperature data (CSV)", type=['csv'])
api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

if not uploaded_file:
    st.info("Upload temperature data CSV to start")
    st.stop()

df = pd.read_csv(uploaded_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

df = calc_moving_avg(df, parallel=False)
df = calc_anomalies(df)
trends = calc_trend(df)
season_stats = calc_city_season_stats(df)

cities = sorted(df['city'].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

city_df = df[df['city'] == selected_city]
slope, r_value, p_value = trends[selected_city]
city_season_stats = season_stats[season_stats['city'] == selected_city]

st.header(f"{selected_city} - Historical Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("mean temperature", f"{city_df['temperature'].mean():.1f}°C")
col2.metric("standard deviation", f"{city_df['temperature'].std():.1f}°C")
col3.metric("minimum temperature", f"{city_df['temperature'].min():.1f}°C")
col4.metric("maximum temperature", f"{city_df['temperature'].max():.1f}°C")

st.info(f"Trend: {slope*365:.2f}°C/year (R² = {r_value**2:.3f})")

st.subheader("Temperature timeseries and moving average")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=city_df['timestamp'], y=city_df['temperature'],
    mode='lines', name='Temperature', line=dict(color='lightblue', width=1)
))
fig1.add_trace(go.Scatter(
    x=city_df['timestamp'], y=city_df['moving_mean'],
    mode='lines', name='Moving Average (30 days)', line=dict(color='black', width=2)
))
fig1.add_trace(go.Scatter(
    x=city_df['timestamp'], y=city_df['lower_bound'],
    mode='lines', name='Lower Bound (-2σ)', line=dict(color='gray', width=1, dash='dash'),
    showlegend=False
))
fig1.add_trace(go.Scatter(
    x=city_df['timestamp'], y=city_df['upper_bound'],
    mode='lines', name='Upper Bound (+2σ)', line=dict(color='gray', width=1, dash='dash'),
    fill='tonexty', fillcolor='rgba(128,128,128,0.2)'
))
fig1.update_layout(height=400, xaxis_title="Date", yaxis_title="Temperature (°C)")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Detected anomalies")

normal = city_df[~city_df['is_anomaly']]
anomalies = city_df[city_df['is_anomaly']]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=normal['timestamp'], y=normal['temperature'],
    mode='markers', name='Normal', marker=dict(size=2, color='gray', opacity=0.5)
))
fig2.add_trace(go.Scatter(
    x=anomalies['timestamp'], y=anomalies['temperature'],
    mode='markers', name=f'Anomaly ({len(anomalies)})', marker=dict(size=5, color='red')
))
fig2.update_layout(height=400, xaxis_title="Date", yaxis_title="Temperature (°C)")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Long-term trend")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=city_df['timestamp'], y=city_df['temperature'],
    mode='markers', name='Temperature', marker=dict(size=2, color='lightblue', opacity=0.3)
))
fig3.add_trace(go.Scatter(
    x=city_df['timestamp'], y=city_df['trend'],
    mode='lines', name=f'Trend: {slope*365:.2f}°C/year', line=dict(color='darkred', width=3)
))
fig3.add_annotation(
    x=0.05, y=0.95, xref='paper', yref='paper',
    text=f"R² = {r_value**2:.3f}<br>p = {p_value:.3f}",
    showarrow=False, bgcolor='white', bordercolor='black', borderwidth=1
)
fig3.update_layout(height=400, xaxis_title="Date", yaxis_title="Temperature (°C)")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Seasonal profiles")

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=city_season_stats['season'], y=city_season_stats['mean_temp'],
    error_y=dict(type='data', array=city_season_stats['std_temp']*2),
    name='Mean ±2σ', marker=dict(color='steelblue')
))
fig4.update_layout(height=350, xaxis_title="Season", yaxis_title="Temperature (°C)")
st.plotly_chart(fig4, use_container_width=True)
    
if not api_key:
    st.info("Enter API key in sidebar to see current weather")
    st.stop()

st.header("Current Weather")

try:
    current = get_current_temp(selected_city, api_key)
    
    if not current or 'temperature' not in current:
        st.error("Invalid API key. Please check your key.")
        st.stop()
    
    season = get_current_season()
    stats = get_season_stats(df, selected_city, season)
    result = check_anomaly(current['temperature'], stats)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Temp", f"{current['temperature']:.1f}°C")
    col2.metric("Feels Like", f"{current['feels_like']:.1f}°C")
    col3.metric("Condition", current['description'].title())
    
    st.info(f"**Historical Range ({season}):** {stats['lower']:.1f}°C - {stats['upper']:.1f}°C")
    
    if result['is_anomaly']:
        st.error(f"**Anomaly detected!**  \nTemperature is {result['status']}")
    else:
        st.success(f"Temperature is {result['status']}")
        
except Exception as e:
    st.error(f"Error: {str(e)}")
