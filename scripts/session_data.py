def load_session_data(year=2025, session_name='British Grand Prix'):
    """
    Loads race lap data for a given year and session name. Raises ValueError if no data is found.
    """
    session_data = get_race_laps(year, session_name)
    if session_data.empty:
        raise ValueError(f"No race data found for {year} {session_name}.")
    return session_data
import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('data') # Cache in /data


def get_race_laps(year: int, session_name: str = 'British Grand Prix') -> pd.DataFrame:
    """
    Get lap data for the Silverstone race for a given year.
    Returns a DataFrame with columns: Driver, LapTime, Sector1Time, Sector2Time, Sector3Time.

    Args:
        year (int): Year of the race.
        session_name (str): Name of the session (default 'British Grand Prix').

    Returns:
        pd.DataFrame: DataFrame with clean lap data.
    """
    if not isinstance(year, int) or year < 1950:
        raise ValueError("Year must be a valid integer (>=1950).")
    try:
        session = fastf1.get_session(year, session_name, 'R')
        session.load()
    except Exception as e:
        raise RuntimeError(f"Error loading session: {e}")

    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    # Remove rows with null values or invalid times
    laps = laps.dropna(subset=["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"])
    # Optionally: filter only valid laps (LapTime > 0)
    laps = laps[laps["LapTime"].apply(lambda x: hasattr(x, "total_seconds") and x.total_seconds() > 0)]

    return laps.reset_index(drop=True)

def get_session_results(year: int, identifier: str, session_name: str = 'British Grand Prix') -> pd.DataFrame:
    if not isinstance(year, int) or year < 1950:
        raise ValueError("Year must be a valid integer (>=1950).")
    try:
        session = fastf1.get_session(year, session_name, identifier)
        session.load()
    except Exception as e:
        raise RuntimeError(f"Error loading session: {e}")
    
    return session.results

