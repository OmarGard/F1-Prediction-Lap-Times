import fastf1
import pandas as pd
import numpy as np
from datetime import timedelta

from fastf1 import Cache
fastf1.Cache.enable_cache('data')  # Cache in /data

# Read the wet performance factors from a CSV file
def read_wet_performance_factors(file_path = 'data/performance/wet_performance_factors.csv') -> pd.DataFrame:
    """
    Reads wet performance factors from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing wet performance factors.
        
    Returns:
        pd.DataFrame: DataFrame containing the wet performance factors.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Wet performance factors data is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Wet performance factors file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading wet performance factors data: {e}")