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
from desc_predict import get_lang_splits, UFODescriptorPredictor

# Download all required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the trained model and scaler
with open("outputs/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("outputs/scaler.pkl", "rb") as f:
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
    fig = px.scatter_map(
        data, lat="latitude", lon="longitude", zoom=3, map_style="open-street-map"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Sightings by Month")
    monthly_counts = data["datetime"].dt.month.value_counts().sort_index()
    fig = px.bar(
        x=monthly_counts.index,
        y=monthly_counts.values,
        labels={"x": "Month", "y": "Number of Sightings"},
    )
    st.plotly_chart(fig, use_container_width=True)

# Text analysis
st.header("Common Words in Sighting Descriptions")
stop_words = set(stopwords.words("english"))
words = [
    word.lower()
    for text in data["comments"].dropna()
    for word in word_tokenize(text)
    if word.isalnum() and word.lower() not in stop_words
]
fdist = FreqDist(words)

# Generate and display wordcloud
wordcloud = WordCloud(
    width=800, height=400, background_color="white"
).generate_from_frequencies(fdist)
st.image(wordcloud.to_array())


st.markdown("---")
st.header("Description Likelihood Predictor Model")
st.markdown("""
This neural network model predicts likely UFO sighting descriptors based on geographic location. It works by:

1. Converting geographic coordinates to geohash strings (spatial encoding)
2. Using an embedding layer to create a dense 32-dimensional vector representation of location
3. Passing this through a neural network to predict a text embedding vector
4. Finding the most similar descriptions from a set of candidates

The model was trained on tens of thousands of UFO sighting reports to learn regional patterns in sighting descriptions.
""")

# Display Neural Network Model!
@st.cache_resource
def load_predictor(model_path):
    try:
        train_df, val_df, test_df = get_lang_splits()
        predictor = UFODescriptorPredictor(model_path=SAVE_PATH)
        if not predictor.is_model_cached():
            raise ValueError("Model not cached.")
            return None
        else:
            return predictor
    except Exception as e:
        st.write(f"Error loading the model: {e}")
        predictor = None


SAVE_PATH = "./outpts/text_predictor_model"
try:
    predictor = load_predictor(SAVE_PATH)
except Exception as e:
    st.write(f"Error loading the model: {e}")
    predictor = None

if predictor:
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Input Location")
        # CSU Fullerton
        lat_default = 33.883094
        lon_default = -117.885215
        lat = st.number_input("Latitude", value=lat_default)
        lon = st.number_input("Longitude", value=lon_default)

    with col4:
        st.subheader("Candidate Descriptions")
        default_descriptions = [
            "bright light hovering in the sky",
            "triangular craft moving silently",
            "pulsating orb changing colors",
            "disk-shaped object with flashing lights",
            "cigar-shaped object moving at high speed",
            "multiple small lights moving erratically",
            "a silent, dark shape against the night sky",
        ]
        candidate_desc_input = st.text_area(
            "Enter candidate descriptions (one per line):",
            value="\n".join(default_descriptions),
            height=210,
        )

    if st.button("Predict Likely Descriptors"):
        candidate_desc = [
            line.strip() for line in candidate_desc_input.split("\n") if line.strip()
        ]

        if not candidate_desc:
            st.warning("Please enter at least one description.")
        elif latitude is not None and longitude is not None:
            try:
                with st.spinner(f"Predicting for ({lat}, {lon})...."):
                    location_embedding = predictor.predict(lat, lon)

                    similarities = predictor.find_similar_descriptions(
                        location_embedding,
                        candidate_desc,
                        top_n=len(candidate_desc),
                    )

                st.subheader("Prediction Results")
                st.write(
                    f"Most likely descriptors for location ({latitude:.4f}, {longitude:.4f}), based on similarity to the candidates provided:"
                )

                # Create a DataFrame for better display
                results_df = pd.DataFrame(
                    similarities, columns=["Description", "Similarity Score"]
                )
                results_df = results_df.sort_values(
                    by="Similarity Score", ascending=False
                ).reset_index(drop=True)

                st.dataframe(results_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

        else:
            st.warning("Please enter valid Latitude and Longitude")

else:
    st.warning("Model could not be loaded.")
