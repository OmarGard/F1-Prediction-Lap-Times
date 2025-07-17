import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('data') # Cache in /data

def get_race_laps(year: int, session_name: str = 'British Grand Prix') -> pd.DataFrame:
    """
    Obtiene los datos de vueltas de la carrera de Silverstone para un año dado.
    Devuelve un DataFrame con las columnas: Driver, LapTime, Sector1Time, Sector2Time, Sector3Time.

    Args:
        year (int): Año de la carrera.
        session_name (str): Nombre de la sesión (por defecto 'British Grand Prix').

    Returns:
        pd.DataFrame: DataFrame con los datos de vueltas limpios.
    """
    if not isinstance(year, int) or year < 1950:
        raise ValueError("El año debe ser un entero válido (>=1950).")
    try:
        session = fastf1.get_session(year, session_name, 'R')
        session.load()
    except Exception as e:
        raise RuntimeError(f"Error cargando la sesión: {e}")

    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
    # Eliminar filas con valores nulos o tiempos no válidos
    laps = laps.dropna(subset=["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"])
    # Opcional: filtrar solo vueltas válidas (LapTime > 0)
    laps = laps[laps["LapTime"].apply(lambda x: hasattr(x, "total_seconds") and x.total_seconds() > 0)]

    return laps.reset_index(drop=True)

def get_session_results(year: int, identifier: str, session_name: str = 'British Grand Prix') -> pd.DataFrame:
    if not isinstance(year, int) or year < 1950:
        raise ValueError("El año debe ser un entero válido (>=1950).")
    try:
        session = fastf1.get_session(year, session_name, identifier)
        session.load()
    except Exception as e:
        raise RuntimeError(f"Error cargando la sesión: {e}")
    
    return session.results

