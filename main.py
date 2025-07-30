
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scripts.session_data import load_session_data, get_target_total_time
from scripts.racepace import get_clean_air_race_pace
from scripts.qualifying import get_qualifying_data
from scripts.team_performance import get_team_performance
from scripts.wet_performance_factor import read_wet_performance_factors
from scripts.utils import (
    get_weather_data,
    convert_times_to_seconds,
    average_sector_times_by_driver,
    add_wet_and_dry_performance_factor,
    add_team_performance_score,
    read_track_data
)

def load_and_prepare_data(year, grand_prix, team_perf_previous_round, team_perf_gp, weather_path, wet_perf_path):
    track_lengths_df = read_track_data()
    track_length_km = track_lengths_df.loc[track_lengths_df['Country'] == 'United Kingdom', ' Length (km)'].values[0]

    session_2024 = load_session_data(2024, 'British Grand Prix')
    session_2024 = convert_times_to_seconds(session_2024)
    sector_times_2024 = average_sector_times_by_driver(session_2024)

    clean_air_race_pace = get_clean_air_race_pace(year, grand_prix)
    qualifying_data = get_qualifying_data(year, grand_prix)
    qualifying_data = qualifying_data.merge(clean_air_race_pace[["Driver", "Median"]], on="Driver", how="left")
    qualifying_data = qualifying_data.rename(columns={"Median": "CleanAirRacePace (s)"})

    weather_data = get_weather_data(weather_path)
    weather_temperature = weather_data.data[0]["temp"]

    wet_and_dry_performance_factors = read_wet_performance_factors(wet_perf_path)
    qualifying_data = add_wet_and_dry_performance_factor(qualifying_data, wet_and_dry_performance_factors)

    team_performance = get_team_performance(year, team_perf_previous_round, team_perf_gp)
    merged = add_team_performance_score(qualifying_data, team_performance, 'Team', 'constructorName')
    merged = merged.merge(sector_times_2024[["Driver", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "TotalSectorTime (s)"]], on="Driver", how="left")
    merged = merged.rename(columns={
        "Sector1Time (s)": "AvgSector1Time (s)",
        "Sector2Time (s)": "AvgSector2Time (s)",
        "Sector3Time (s)": "AvgSector3Time (s)",
        "TotalSectorTime (s)": "AvgTotalSectorTime (s)",
        "wet_performance_factor": "WetPerformanceFactor",
        "dry_performance_factor": "DryPerformanceFactor",
        "Best_Time_Seconds": "QualifyingTime"
    })

    merged["TrackTemperature"] = weather_temperature
    merged["RainIntensity"] = weather_data.data[0].get("rain", {}).get("3h", 0)

    merged["Velocity (km/h)"] = (track_length_km * 3600) / merged["QualifyingTime"]
    return merged

def get_features_and_target(merged, year, grand_prix):
    X = merged[[
        "Velocity (km/h)", "TrackTemperature", "TeamPerformanceScore", "RainIntensity", "CleanAirRacePace (s)", "WetPerformanceFactor", "DryPerformanceFactor"
    ]]
    target_total_times = get_target_total_time(year, 'R', grand_prix)

    track_data_df = read_track_data()
    track_length_km = track_data_df.loc[track_data_df['Country'] == 'United Kingdom', ' Length (km)'].values[0]
    num_laps = track_data_df.loc[track_data_df['Country'] == 'United Kingdom', ' Laps'].values[0]

    target_total_times = target_total_times.copy()  # Evita el warning
    target_total_times.loc[:, "Velocity (km/h)"] = (track_length_km * num_laps * 3600) / target_total_times["TotalTime (s)"]
    
    y = target_total_times["Velocity (km/h)"]

    return X, y

def train_and_predict(X, y, merged):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=10, random_state=37)
    model.fit(X_train, y_train.values.ravel())
    merged["PredictedVelocity (km/h)"] = model.predict(X_imputed)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, merged, mae, X_test, y_test, y_pred

def plot_results(final_results, model, X):
    print("\n🏁 Predicted 2025 British Grand Prix Winner 🏁\n")
    print(final_results[["Driver", "PredictedVelocity (km/h)"]])

    plt.figure(figsize=(12, 8))
    plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedVelocity (km/h)"])
    for i, driver in enumerate(final_results["Driver"]):
        plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedVelocity (km/h)"].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')
    plt.xlabel("clean air race pace (s)")
    plt.ylabel("predicted velocity (km/h)")
    plt.title("effect of clean air race pace on predicted velocity")
    plt.tight_layout()
    plt.show()

    feature_importance = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8,5))
    plt.barh(features, feature_importance, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance in Race Time Prediction")
    plt.tight_layout()
    plt.show()

def main():
    year = 2025
    grand_prix = 'British Grand Prix'
    team_perf_previous_round = 11
    team_perf_gp = 'Austria'
    weather_path = "data/weather/great_britain_2025_weather.json"
    wet_perf_path = 'data/performance/wet_performance_factors.csv'

    merged = load_and_prepare_data(year, grand_prix, team_perf_previous_round, team_perf_gp, weather_path, wet_perf_path)
    X, y = get_features_and_target(merged, year, grand_prix)
    model, merged, mae, X_test, y_test, y_pred = train_and_predict(X, y, merged)
    final_results = merged.sort_values("PredictedVelocity (km/h)", ascending=False).reset_index(drop=True)
    plot_results(final_results, model, X)
    print(f"Model Error (MAE): {mae:.2f} km/h")

if __name__ == "__main__":
    main()
