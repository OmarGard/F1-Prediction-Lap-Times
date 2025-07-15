import pandas as pd
import os
from dotenv import load_dotenv
from get_silverstone_data import get_race_laps
from racepace import get_clean_air_race_pace
from qualifying import get_qualifying_data
from teamPerformance import getTeamPerformance
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

def addWetPerformanceFactor(qualifying_data):
    qualifying_data["WetPerformanceFactor"] = 1.3  # Example factor, adjust as needed
    qualifying_data["QualifyingTime"] = qualifying_data["Best_Time_Seconds"] * qualifying_data["WetPerformanceFactor"]
    return qualifying_data 

# session_2025 = load_session_data(2025, 'British Grand Prix')
# session_2025 = convert_times_to_seconds(session_2025)
# sector_times_2025 = average_sector_times_by_driver(session_2025)
# clean_air_race_pace = get_clean_air_race_pace(2025, 'British Grand Prix')
# qualifying_data = get_qualifying_data(2025, 'British Grand Prix')
# weather_data = get_weather_data(WEATHER_API_KEY, forecast_time="2025-07-06 13:00:00")
# if weather_data.data.weather.main == "Rain" or weather_data.data.weather.main == "Drizzle" or weather_data.data.weather.main == "Thunderstorm":
#     qualifying_data = addWetPerformanceFactor(qualifying_data)
team_performance = getTeamPerformance(2025, 12, 'Great_Britain')
print(team_performance)

