import pandas as pd
import os
from get_silverstone_data import get_race_laps
from racepace import get_clean_air_race_pace

# Definir rutas base y de resultados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data')

# Usa la función get_race_laps de get_silverstone_data.py para obtener los datos de la carrera silverstone 2024
def load_silverstone_data():
    session_data = get_race_laps(2024, session_name='British Grand Prix')
    if session_data.empty:
        raise ValueError("No se encontraron datos de la carrera para 2024.")
    return session_data

def convert_times_to_seconds(df):
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[f"{col} (s)"] = df[col].dt.total_seconds()
    return df

def average_sector_times_by_driver(df):
    new_df = df.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    new_df["TotalSectorTime (s)"] = ( new_df["Sector1Time (s)"] + new_df["Sector2Time (s)"] + new_df["Sector3Time (s)"])
    return new_df

session_2024 = load_silverstone_data()
session_2024 = convert_times_to_seconds(session_2024)
sector_times_2024 = average_sector_times_by_driver(session_2024)
clean_air_race_pace = get_clean_air_race_pace(2025, 'British Grand Prix')

print(clean_air_race_pace)

