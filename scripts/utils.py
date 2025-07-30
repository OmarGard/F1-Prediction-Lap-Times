import pandas as pd
from fuzzywuzzy import process


def get_weather_data(weather_file_path):
    if not weather_file_path:
        raise ValueError("The weather data file cannot be empty.")
    try:
        weather_data = pd.read_json(weather_file_path)
        if weather_data.empty:
            raise ValueError("The weather data is empty.")
        return weather_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Weather data file not found: {weather_file_path}")
    except Exception as e:
        raise Exception(f"Error loading weather data: {e}")

# --- PIPELINE FUNCTIONS THAT CAN BE REUSED ---
def fill_missing_total_times(y):
    """
    Fills NaT values in the 'TotalTime (s)' column by adding the average gap to the maximum known time.
    """
    max_time = y["TotalTime (s)"].max()
    avg_distance = y["TotalTime (s)"].diff().mean()
    for idx in y[y["TotalTime (s)"].isna()].index:
        max_time += avg_distance
        y.at[idx, "TotalTime (s)"] = max_time
    return y

def convert_times_to_seconds(df):
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[f"{col} (s)"] = df[col].dt.total_seconds()
    return df

def average_sector_times_by_driver(df):
    new_df = df.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    new_df["TotalSectorTime (s)"] = (
        new_df["Sector1Time (s)"] + new_df["Sector2Time (s)"] + new_df["Sector3Time (s)"])
    return new_df

def add_wet_and_dry_performance_factor(qualifying_data, wet_and_dry_performance_factors):
    qualifying_data = qualifying_data.merge(
        wet_and_dry_performance_factors[["Driver", "wet_performance_factor", "dry_performance_factor"]],
        on="Driver",
        how="left"
    )
    return qualifying_data

def get_fuzzy_matches(df1, df2, key1, key2, threshold=80):
    matches = []
    for value in df1[key1].unique():
        result = process.extractOne(value, df2[key2], score_cutoff=threshold)
        if result:
            match = result[0]
            matches.append((value, match))
    return pd.DataFrame(matches, columns=[key1, key2])

def add_team_performance_score(df1, df2, key1, key2, threshold=80):
    matched_df = get_fuzzy_matches(df1, df2, key1, key2, threshold)
    merged_df = df1.merge(matched_df, on=key1, how='left')
    merged_df = merged_df.merge(df2[[key2, 'points']], on=key2, how='left')
    merged_df = merged_df.drop(columns=[key2])
    merged_df = merged_df.rename(columns={'points': 'TeamPerformanceScore'})
    return merged_df

def read_track_data(file_path='data/tracks/tracks_length.csv'):
    """
    Reads the track data file and returns a DataFrame.
    Args:
        file_path (str): Path to the track data CSV file.
    Returns:
        pd.DataFrame: DataFrame with track information and their data.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The track data file is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Track data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading track data file: {e}")
