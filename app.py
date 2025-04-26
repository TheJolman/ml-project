
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import pickle
from data_loader import load_ufo_data

# Download all required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and scaler
with open('outputs/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('outputs/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load and prepare data
data = load_ufo_data()

st.set_page_config(page_title="UFO Sightings Analysis", layout="wide")
st.title("UFO Sightings Analysis and Prediction")

# Sidebar for predictions
st.sidebar.header("Predict UFO Sightings")
latitude = st.sidebar.slider("Latitude", 25.0, 50.0, 37.5)
longitude = st.sidebar.slider("Longitude", -125.0, -65.0, -95.0)
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 3)
hour = st.sidebar.slider("Hour", 0, 23, 12)

# Make prediction
features = np.array([[latitude, longitude, month, day_of_week, hour]])
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]

st.sidebar.markdown(f"### Predicted Sightings\n{prediction:.2f}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Sightings Distribution")
    fig = px.scatter_mapbox(data, 
                           lat='latitude', 
                           lon='longitude',
                           zoom=3,
                           mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Sightings by Month")
    monthly_counts = data['datetime'].dt.month.value_counts().sort_index()
    fig = px.bar(x=monthly_counts.index, 
                 y=monthly_counts.values,
                 labels={'x': 'Month', 'y': 'Number of Sightings'})
    st.plotly_chart(fig, use_container_width=True)

# Text analysis
st.header("Common Words in Sighting Descriptions")
stop_words = set(stopwords.words('english'))
words = [word.lower() for text in data['comments'].dropna() 
         for word in word_tokenize(text) 
         if word.isalnum() and word.lower() not in stop_words]
fdist = FreqDist(words)

# Generate and display wordcloud
wordcloud = WordCloud(width=800, height=400,
                     background_color='white').generate_from_frequencies(fdist)
st.image(wordcloud.to_array())
