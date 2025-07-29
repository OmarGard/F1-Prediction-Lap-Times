from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scripts.session_data import get_session_results, load_session_data, get_target_total_time
from scripts.racepace import get_clean_air_race_pace
from scripts.qualifying import get_qualifying_data
from scripts.team_performance import get_team_performance
from scripts.wet_performance_factor import read_wet_performance_factors
from scripts.utils import (
    get_weather_data,
    convert_times_to_seconds,
    average_sector_times_by_driver,
    add_wet_and_dry_performance_factor,
    add_team_performance_score
)

# --- PREDICTION PIPELINE AND VISUALIZATION ---

# Data loading and processing
session_2024 = load_session_data(2024, 'British Grand Prix')
session_2024 = convert_times_to_seconds(session_2024)
sector_times_2024 = average_sector_times_by_driver(session_2024)

clean_air_race_pace = get_clean_air_race_pace(2025, 'British Grand Prix')
qualifying_data = get_qualifying_data(2025, 'British Grand Prix')
qualifying_data = qualifying_data.merge(clean_air_race_pace[["Driver", "Median"]], on="Driver", how="left")
qualifying_data = qualifying_data.rename(columns={"Median": "CleanAirRacePace (s)"})

weather_data = get_weather_data("data/weather/great_britain_2025_weather.json")
weather_forecast = weather_data.data[0]["weather"][0]["main"]
weather_temperature = weather_data.data[0]["temp"]

wet_and_dry_performance_factors = read_wet_performance_factors('data/performance/wet_performance_factors.csv')
qualifying_data = add_wet_and_dry_performance_factor(qualifying_data, wet_and_dry_performance_factors)

team_performance = get_team_performance(2025, 11, 'Austria')
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

# Features and target
X = merged[[
    "QualifyingTime", "TrackTemperature", "TeamPerformanceScore", "RainIntensity", "CleanAirRacePace (s)", "WetPerformanceFactor", "DryPerformanceFactor"
]]

y = get_target_total_time(2025, 'R', 'British Grand Prix')

# Imputation and model
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=10, random_state=37)
model.fit(X_train, y_train.values.ravel())
merged["PredictedTime"] = model.predict(X_imputed)


# Results and visualization
final_results = merged.sort_values("PredictedTime")
print("\n🏁 Predicted 2025 British Grand Prix Winner 🏁\n")
print(final_results[["Driver", "PredictedTime"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Effect of clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedTime"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedTime"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("clean air race pace (s)")
plt.ylabel("predicted race time (s)")
plt.title("effect of clean air race pace on predicted race results")
plt.tight_layout()
plt.show()

# Feature importance
feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()
