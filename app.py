import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from lime.lime_text import LimeTextExplainer

# Add a hero image
st.image('images/hero.jpg', use_column_width=True)

# Load time series data
df_resampled = pd.read_csv('data/reviews_processed_tsa.csv')
df_resampled.set_index('date', inplace=True)

# Add app title
st.title('Time Series Analysis Customer Review for Sandbar')

# User input for the time frame selection and sentiment analysis (positive/negative)
st.subheader('Select a Time Frame')
time_frame = st.slider('Time Frame (Months)',
                       min_value=1,
                       max_value=(len(df_resampled)),
                       step=1)

# Resample data according to the user-selected time frame
resampled_data = df_resampled['stars'].rolling(window=time_frame).mean()

# Plot the time series data
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_resampled.index,
    y=df_resampled['stars'],
    mode='lines',
    name='Monthly Average'
))

# Add moving averages to the plot as lines on top
fig.add_trace(go.Scatter(
    x=df_resampled.index,
    y=resampled_data,
    mode='lines',
    name=f'{time_frame}-Month Moving Average'
))

# Add title and labels to the plot
fig.update_layout(
    title = f'Average Star Rating Over Time with {time_frame}-Monthly Moving Average',
    xaxis_title = 'Time',
    yaxis_title = 'Average Star Rating',
)

# Show the plot
st.plotly_chart(fig, use_container_width=True)