from data_loader import load_ufo_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import pickle
import time


def prepare_data():
    print("\nPreparing data...")

    print("Loading UFO sightings data...")
    data = load_ufo_data()

    print("Creating location grid...")
    lat_bins = np.linspace(25, 50, 50)  # US latitude range
    lon_bins = np.linspace(-125, -65, 60)  # US longitude range

    print("Binning locations...")
    data["lat_bin"] = pd.cut(data["latitude"], lat_bins, labels=lat_bins[:-1])
    data["lon_bin"] = pd.cut(data["longitude"], lon_bins, labels=lon_bins[:-1])

    print("Extracting time features...")
    data["month"] = data["datetime"].dt.month
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["hour"] = data["datetime"].dt.hour

    print("Aggregating sightings...")
    location_counts = (
        data.groupby(["lat_bin", "lon_bin", "month", "day_of_week", "hour"])
        .size()
        .reset_index(name="sighting_count")
    )

    X = location_counts[["lat_bin", "lon_bin", "month", "day_of_week", "hour"]]
    y = location_counts["sighting_count"]

    print(f"Prepared {len(X)} samples with {X.shape[1]} features")
    return X, y


def train_model():
    start_time = time.time()

    # Prepare the data
    X, y = prepare_data()

    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    print("\nTraining Random Forest model...")
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        verbose=1,  # Enable built-in verbosity
    )

    rf_regressor.fit(X_train, y_train)

    print("\nMaking predictions...")
    y_pred = rf_regressor.predict(X_test)

    # Print evaluation metrics
    print("\nModel Performance:")
    print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    print("\nSaving model and scaler...")
    with open("outputs/rf_model.pkl", "wb") as f:
        pickle.dump(rf_regressor, f)
    with open("outputs/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")

    return rf_regressor, scaler


if __name__ == "__main__":
    train_model()
