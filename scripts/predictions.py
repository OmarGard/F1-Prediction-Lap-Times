import pandas as pd
from get_silverstone_data import get_race_laps
from racepace import get_clean_air_race_pace
from qualifying import get_qualifying_data

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

session_2025 = load_session_data(2025, 'British Grand Prix')
session_2025 = convert_times_to_seconds(session_2025)
sector_times_2025 = average_sector_times_by_driver(session_2025)
clean_air_race_pace = get_clean_air_race_pace(2025, 'British Grand Prix')
qualifying_data = get_qualifying_data(2025, 'British Grand Prix')

print(qualifying_data)

