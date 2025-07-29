import pandas as pd

def get_team_performance(season, round, round_name):
    """
    Get team performance for a specific season and round.
    
    Args:
        season (int): The year of the season.
        round (str): The Grand Prix round.
        
    Returns:
        pd.DataFrame: A DataFrame with team performance data.
    """
    try:
        file_path = f'data/standings/constructor_standings_{season}_Round_{round}_{round_name}.csv'
        team_performance = pd.read_csv(file_path)
        if team_performance.empty:
            raise ValueError("Team performance data is empty.")
        return team_performance
    except FileNotFoundError:
        raise FileNotFoundError(f"Team performance file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading team performance data: {e}")
    
