import pandas as pd
import kagglehub


def load_ufo_data() -> pd.DataFrame:
    # Download latest version
    path = kagglehub.dataset_download("sahityasetu/ufo-sightings")

    # Load and clean the data
    data = pd.read_csv(
        f"{path}/ufo_sightings_scrubbed.csv",
        dtype={
            "duration (seconds)": str,
            "latitude": str,
        },
    )
    data.rename(columns={data.columns[10]: "longitude"}, inplace=True)
    data["duration (seconds)"] = (
        data["duration (seconds)"]
        .replace(to_replace=r"[^0-9\.]", value="", regex=True)
        .astype(float)
    )
    data["latitude"] = data["latitude"].replace("33q.200088", "33.200088")
    data["latitude"] = data["latitude"].astype(float)

    data["date posted"] = pd.to_datetime(data["date posted"], errors="coerce")
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")

    return data


if __name__ == "__main__":
    data = load_ufo_data()
    print(data.info())
    print(data.head())
