from matplotlib import pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from get_silverstone_data import get_race_laps
from racepace import get_clean_air_race_pace
from qualifying import get_qualifying_data
from teamPerformance import get_team_performance
from fuzzywuzzy import process

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_weather_data(weather_file_path):
    if not weather_file_path:
        raise ValueError("El archivo de datos del clima no puede estar vacío.")
    
    try:
        # read the weather data from the json file inside ../data folder
        weather_data = pd.read_json(weather_file_path)
        if weather_data.empty:
            raise ValueError("Los datos del clima están vacíos.")
        return weather_data
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de datos del clima: {weather_file_path}")
    except Exception as e:
        raise Exception(f"Error al cargar los datos del clima: {e}")

def load_session_data(year=2025, session_name='British Grand Prix'):
    session_data = get_race_laps(year, session_name)
    if session_data.empty:
        raise ValueError(f"No se encontraron datos de la carrera para {year} {session_name}.")
    return session_data

def convert_times_to_seconds(df):
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[f"{col} (s)"] = df[col].dt.total_seconds()
    return df

def average_sector_times_by_driver(df):
    new_df = df.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    new_df["TotalSectorTime (s)"] = ( new_df["Sector1Time (s)"] + new_df["Sector2Time (s)"] + new_df["Sector3Time (s)"])
    return new_df

def add_wet_performance_factor(qualifying_data):
    qualifying_data["WetPerformanceFactor"] = 1.3  # Example factor, adjust as needed
    qualifying_data["QualifyingTime"] = qualifying_data["Best_Time_Seconds"] * qualifying_data["WetPerformanceFactor"]
    return qualifying_data 

def get_fuzzy_matches(df1, df2, key1, key2, threshold=80):
    """
    Returns a DataFrame with fuzzy matches between key1 in df1 and key2 in df2.
    """
    matches = []
    for value in df1[key1].unique():
        result = process.extractOne(value, df2[key2], score_cutoff=threshold)
        if result:
            match = result[0]
            matches.append((value, match))
    return pd.DataFrame(matches, columns=[key1, key2])

def add_team_performance_score(df1, df2, key1, key2, threshold=80):
    """
    Adds the 'points' column from df2 to df1 based on fuzzy matches and renames it to 'TeamPerformanceScore'.
    """
    matched_df = get_fuzzy_matches(df1, df2, key1, key2, threshold)
    merged_df = df1.merge(matched_df, on=key1, how='left')
    merged_df = merged_df.merge(df2[[key2, 'points']], on=key2, how='left')
    merged_df = merged_df.drop(columns=[key2])
    merged_df = merged_df.rename(columns={'points': 'TeamPerformanceScore'})
    return merged_df

# Load 2024 session data and average sector times
session_2024 = load_session_data(2024, 'British Grand Prix')
session_2024 = convert_times_to_seconds(session_2024)
sector_times_2024 = average_sector_times_by_driver(session_2024)

# Load clean air race pace and qualifying data
clean_air_race_pace = get_clean_air_race_pace(2025, 'British Grand Prix')
qualifying_data = get_qualifying_data(2025, 'British Grand Prix')

# Add clean air race pace to qualifying data
qualifying_data = qualifying_data.merge(clean_air_race_pace[["Driver", "Mean"]], on="Driver", how="left")
qualifying_data = qualifying_data.rename(columns={"Mean": "CleanAirRacePace (s)"})

# Load weather data and adjust qualifying data if necessary
weather_data = get_weather_data("data/weather/great_britain_2025_weather.json")
weather_forecast = weather_data.data[0]["weather"][0]["main"]
weather_temperature = weather_data.data[0]["temp"]
if weather_forecast == "Rain" or weather_forecast == "Drizzle" or weather_forecast == "Thunderstorm":
    qualifying_data = add_wet_performance_factor(qualifying_data)

# Load team performance data and merge with qualifying data
team_performance = get_team_performance(2025, 11, 'Austria')
merged = add_team_performance_score(qualifying_data, team_performance, 'Team', 'constructorName')

# Merge sector times with qualifying data
merged = merged.merge(sector_times_2024[["Driver", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "TotalSectorTime (s)"]], on="Driver", how="left")

# Add temperature to the merged DataFrame
merged["TrackTemperature"] = weather_temperature
merged["RainProbability"] = 0.8

# Filter out drivers not present in the 2025 and 2024 session data
valid_drivers = merged["Driver"].isin(session_2024["Driver"].unique())
merged = merged[valid_drivers]

# define features (X) and target (y)
X = merged[[
    "QualifyingTime", "TrackTemperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"
]]
y = session_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged["Driver"])

# impute missing values for features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

# train gradient boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
model.fit(X_train, y_train)
merged["PredictedRaceTime (s)"] = model.predict(X_imputed)

# sort the results to find the predicted winner
final_results = merged.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 British Grand Prix Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# plot effect of clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("clean air race pace (s)")
plt.ylabel("predicted race time (s)")
plt.title("effect of clean air race pace on predicted race results")
plt.tight_layout()
plt.show()

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()


# sort results and get top 3
final_results = merged.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted in the Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")

