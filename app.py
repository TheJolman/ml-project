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
try:
    with open("outputs/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Model file not found. Please ensure the model is trained and saved correctly."
    )
    model = None

try:
    with open("outputs/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Scaler file not found. Please ensure the model is trained and saved correctly."
    )
    scaler = None

# Load and prepare data

try:
    data = load_ufo_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = None

st.set_page_config(page_title="UFO Sightings Analysis", layout="wide")
st.title("UFO Sightings Analysis and Prediction")

# Sidebar for predictions
st.sidebar.header("Predict UFO Sightings")
st.markdown("""
This Random Forest model predicts the number of UFO sightings based on location and time parameters. It works by:

1. Binning locations into a grid across US latitude and longitude
2. Extracting time-based features (month, day of week, hour)
3. Using a Random Forest regressor trained on historical data
4. Scaling inputs with StandardScaler for better performance
""")
latitude = st.sidebar.slider("Latitude", 25.0, 50.0, 37.5)
longitude = st.sidebar.slider("Longitude", -125.0, -65.0, -95.0)
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 3)
hour = st.sidebar.slider("Hour", 0, 23, 12)

if not data.empty and model and scaler:
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

else:
    if not data:
        st.error("Data not loaded. Please check the data source.")
    elif not model:
        st.error(
            "Random Forest model not loaded. Please check that the model file exists."
        )
    elif not scaler:
        st.error("Scaler not loaded. Please check that the scaler file exists.")


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


# Path to saved model
SAVE_PATH = "./outpts/text_predictor_model"


# Display Neural Network Model!
@st.cache_resource
def load_predictor(model_path):
    try:
        # Get data splits
        train_df, val_df, test_df = get_lang_splits()

        # Initialize predictor with model path
        predictor = UFODescriptorPredictor(model_path=model_path)

        # Verify model is available
        if not predictor.is_model_cached():
            raise ValueError("Model files not found in specified path.")

        return predictor
    except ValueError as e:
        st.error(f"Model validation error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        return None


# Load the predictor
predictor = load_predictor(SAVE_PATH)

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
        # Process description input
        candidate_desc = [
            line.strip() for line in candidate_desc_input.split("\n") if line.strip()
        ]

        # Input validation
        if not candidate_desc:
            st.warning("Please enter at least one description.")

        if lat is None or lon is None:
            st.warning("Please enter valid latitude and longitude values.")

        try:
            with st.spinner(f"Predicting for ({lat}, {lon})..."):
                # Get embedding for the location
                location_embedding = predictor.predict(lat, lon)

                # Find similar descriptions
                similarities = predictor.find_similar_descriptions(
                    location_embedding,
                    candidate_desc,
                    top_n=len(candidate_desc),
                )

            # Display results
            st.subheader("Prediction Results")
            st.write(
                f"Most likely descriptors for location ({lat:.4f}, {lon:.4f}), based on similarity to the candidates provided:"
            )

            # Create a DataFrame for better display
            results_df = pd.DataFrame(
                similarities, columns=["Description", "Similarity Score"]
            )
            results_df = results_df.sort_values(
                by="Similarity Score", ascending=False
            ).reset_index(drop=True)

            st.dataframe(results_df, use_container_width=True)

        except ValueError as e:
            st.error(f"Invalid input values: {e}")
        except TypeError as e:
            st.error(f"Type error in prediction: {e}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback

            st.error(f"Details: {traceback.format_exc()}")

else:
    st.warning("Model could not be loaded.")

# same thing but compared against test set
st.markdown("---")

try:
    with open("outputs/test_df_results.pkl", "rb") as f:
        test_df = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Test DataFrame file not found. Please ensure the test results are saved correctly."
    )
    test_df = None

if not test_df.empty:
    st.header("Test Set Predictions")
    st.markdown(
        "This section displays the similarity scores for candidate descriptions on the test set. "
        "Below are the candidate descriptions (same for every row), followed by a table of similarity scores per test sample."
    )
    preds = test_df.get("predicted_top_n", [])
    if not preds.empty:
        # Get the candidate descriptions (first elements of tuples) from the first row
        candidates = [desc for desc, _ in preds[1]]
        # Display candidate descriptions for reference
        st.subheader("Candidate Descriptions")
        for idx, desc in enumerate(candidates, start=1):
            st.text(f"{idx}. {desc}")
        # Build a DataFrame of similarity scores (second elements)
        scores = []
        for row in preds:
            # convert score strings to float
            scores.append([float(score) for _, score in row])
        scores_df = pd.DataFrame(scores, columns=candidates)
        # Insert the true descriptions as the leftmost column
        scores_df.insert(0, "true_description", test_df["true_description"].reset_index(drop=True))
        st.subheader("Similarity Scores per Test Sample")
        st.dataframe(scores_df, use_container_width=True)
    else:
        st.info("No predicted similarity scores available for the test set.")
