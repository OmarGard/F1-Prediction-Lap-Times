import pandas as pd
def get_team_performance(season, round, round_name):
    """
    Obtiene el rendimiento del equipo para una temporada y ronda específicas.
    
    Args:
        season (int): El año de la temporada.
        round (str): La ronda del Gran Premio.
        
    Returns:
        pd.DataFrame: Un DataFrame con el rendimiento del equipo.
    """
    try:
        file_path = f'data/standings/constructor_standings_{season}_Round_{round}_{round_name}.csv'
        team_performance = pd.read_csv(file_path)
        if team_performance.empty:
            raise ValueError("Los datos del rendimiento del equipo están vacíos.")
        return team_performance
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de rendimiento del equipo: {file_path}")
    except Exception as e:
        raise Exception(f"Error al cargar los datos del rendimiento del equipo: {e}")
    
