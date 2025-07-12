import fastf1
import pandas as pd
import numpy as np
from datetime import timedelta

from fastf1 import Cache
fastf1.Cache.enable_cache('data')  # Cache in /data

def is_clean_air_lap(lap, gap_threshold=3.0):
    """
    Determine if a lap is in clean air using DriverAheadTime telemetry data
    
    Args:
        lap: The lap to check
        gap_threshold: Minimum gap in seconds to be considered clean air
    
    Returns:
        bool: True if lap is in clean air
    """
    # Use DriverAheadTime from telemetry - much more accurate than time estimation
    driver_ahead_time = lap.get('DriverAheadTime', None)
    
    # If no data available, assume clean air
    if pd.isna(driver_ahead_time) or driver_ahead_time is None:
        return True
    
    # Convert to seconds if it's a timedelta
    if hasattr(driver_ahead_time, 'total_seconds'):
        gap_seconds = driver_ahead_time.total_seconds()
    else:
        gap_seconds = float(driver_ahead_time)
    
    return gap_seconds >= gap_threshold

def extract_race_pace_laps(session_laps, min_stint_length=5):
    """
    Extract race pace representative laps from FP2
    
    Args:
        session_laps: All laps from the session
        min_stint_length: Minimum number of laps in a stint to be considered
    
    Returns:
        DataFrame: Filtered laps suitable for race pace analysis
    """
    race_pace_laps = []
    
    for driver in session_laps['Driver'].unique():
        driver_laps = session_laps[session_laps['Driver'] == driver].copy()
        driver_laps = driver_laps.sort_values('LapStartTime')
        
        # Remove outliers (laps that are too slow or too fast)
        if len(driver_laps) > 0:
            lap_times = driver_laps['LapTime'].dt.total_seconds()
            q1 = np.percentile(lap_times, 25)
            q3 = np.percentile(lap_times, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            driver_laps = driver_laps[
                (lap_times >= lower_bound) & 
                (lap_times <= upper_bound)
            ].copy()  # Add .copy() to avoid warnings
        
        # Group consecutive laps into stints
        driver_laps = driver_laps.copy()  # Ensure we have a copy
        driver_laps.loc[:, 'LapNumber'] = range(1, len(driver_laps) + 1)
        driver_laps.loc[:, 'TimeDiff'] = driver_laps['LapStartTime'].diff()
        driver_laps.loc[:, 'NewStint'] = driver_laps['TimeDiff'] > timedelta(minutes=5)
        driver_laps.loc[:, 'StintNumber'] = driver_laps['NewStint'].cumsum()
        
        # Filter stints with minimum length
        stint_lengths = driver_laps.groupby('StintNumber').size()
        valid_stints = stint_lengths[stint_lengths >= min_stint_length].index
        
        driver_race_laps = driver_laps[driver_laps['StintNumber'].isin(valid_stints)].copy()
        
        # Only keep laps from the middle portion of each stint (exclude first 2 and last 2 laps)
        for stint in valid_stints:
            stint_laps = driver_race_laps[driver_race_laps['StintNumber'] == stint].copy()
            if len(stint_laps) > 4:  # Need at least 5 laps to exclude first and last 2
                stint_laps = stint_laps.iloc[2:-2].copy()  # Remove first 2 and last 2 laps
                race_pace_laps.append(stint_laps)
    
    if race_pace_laps:
        return pd.concat(race_pace_laps, ignore_index=True)
    else:
        return pd.DataFrame()

def get_clean_air_race_pace(year, session_name):
    """
    Devuelve el clean air race pace en segundos para todos los pilotos en la sesión especificada.
    
    Args:
        year (int): Año de la sesión
        session_name (str): Nombre de la sesión, por ejemplo 'FP2'
    
    Returns:
        DataFrame: Race pace por piloto con columnas ['Driver', 'Laps', 'Mean', 'Median', 'Std', 'Best', 'Worst']
    """
    # Cargar la sesión
    session = fastf1.get_session(year, session_name, 'FP2')
    session.load(laps=True, telemetry=True, weather=False)
    laps = session.laps

    # Filtrar laps válidas
    valid_laps = laps[
        (laps['LapTime'].notna()) & 
        (laps['IsAccurate'] == True) &
        (~laps['TrackStatus'].isin(['4', '5', '6', '7']))  # Exclude SC, VSC, Red flag conditions
    ].copy()

    # Extract race pace laps
    race_pace_laps = extract_race_pace_laps(valid_laps)

    # Filtrar por clean air usando DriverAheadTime
    clean_air_laps = []
    for idx, lap in race_pace_laps.iterrows():
        if is_clean_air_lap(lap, gap_threshold=3.0):
            clean_air_laps.append(lap)
    if clean_air_laps:
        clean_air_df = pd.DataFrame(clean_air_laps)
        analysis_df = clean_air_df.copy()
        analysis_df.loc[:, 'LapTimeSeconds'] = analysis_df['LapTime'].dt.total_seconds()
        pace_analysis = analysis_df.groupby('Driver')['LapTimeSeconds'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        pace_analysis.columns = ['Laps', 'Mean', 'Median', 'Std', 'Best', 'Worst']
        pace_analysis = pace_analysis.sort_values('Median')
        return pace_analysis.reset_index()
    else:
        return pd.DataFrame(columns=['Driver', 'Laps', 'Mean', 'Median', 'Std', 'Best', 'Worst'])