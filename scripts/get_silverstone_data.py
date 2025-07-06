import fastf1
from fastf1 import plotting
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data')
fastf1.Cache.enable_cache(CACHE_DIR) # Cache in /data

def get_race_laps(year):
    session = fastf1.get_session(year, 'British Grand Prix', 'R')
    session.load()
    
    laps = session.laps
    drivers = session.drivers
    results = session.results
    driver_data = []

    for drv in drivers:
        dr_laps = laps.pick_drivers(drv)
        if dr_laps.empty:
            continue

        avg_pace = dr_laps['LapTime'].mean()
        # Obtener la secuencia de compuestos usados por el piloto durante la carrera
        compound_sequence = list(dr_laps['Compound'].dropna().unique())

        result = {
            'Race': f"{year} Silverstone",
            'Driver': drv,
            'Team': session.get_driver(drv)['TeamName'],
            'Grid Pos': results.loc[results['DriverNumber'] == drv, 'GridPosition'].values[0],
            'Avg Lap Time': avg_pace.total_seconds(),
            'Tire Strategy': compound_sequence,
            'Result Pos': dr_laps.iloc[-1]['Position']
        }

        driver_data.append(result)

    return pd.DataFrame(driver_data)

if __name__ == "__main__":
    df = get_race_laps(2024)
    df.to_csv(os.path.join(CACHE_DIR, 'silverstone_2024.csv'), index=False)
    print(df.head())
