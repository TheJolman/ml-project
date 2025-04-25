
from data_loader import load_ufo_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def prepare_data():
    # Load the data
    data = load_ufo_data()
    
    # Clean and prepare features
    data = data.dropna(subset=['shape', 'latitude', 'longitude', 'duration (seconds)'])
    
    # Create features
    X = data[['latitude', 'longitude', 'duration (seconds)']]
    y = data['shape']  # Predicting UFO shape based on location and duration
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def train_model():
    # Prepare the data
    X, y, label_encoder = prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Print evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and label encoder
    with open('outputs/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    with open('outputs/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return rf_classifier, label_encoder

if __name__ == "__main__":
    train_model()
