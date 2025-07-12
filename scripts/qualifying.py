import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Get the base directory path
fastf1.Cache.enable_cache('data')  # Cache in /data

def get_qualifying_data(year, session_name):
    """
    Get qualifying data for a specific F1 Grand Prix session
    
    Parameters:
    year (int): The year of the season (e.g., 2024, 2023)
    session_name (str): Name of the Grand Prix (e.g., 'Monaco', 'British', 'Abu Dhabi')
    
    Returns:
    pandas.DataFrame: Qualifying results with times in seconds
    """
    
    try:
        # Load the qualifying session
        print(f"Loading {year} {session_name} Grand Prix qualifying data...")
        session = fastf1.get_session(year, session_name, 'Q')
        session.load()
        
        # Get qualifying results
        qualifying_results = session.results
        
        # Create a clean dataframe with relevant information
        df = pd.DataFrame({
            'Position': qualifying_results['Position'],
            'Driver': qualifying_results['Abbreviation'],
            'Full_Name': qualifying_results['FullName'],
            'Team': qualifying_results['TeamName'],
            'Q1_Time_Seconds': qualifying_results['Q1'].dt.total_seconds(),
            'Q2_Time_Seconds': qualifying_results['Q2'].dt.total_seconds(),
            'Q3_Time_Seconds': qualifying_results['Q3'].dt.total_seconds(),
            'Best_Time_Seconds': qualifying_results['Q1'].dt.total_seconds()
        })
        
        # Fill the Best_Time_Seconds with the best qualifying time for each driver
        for idx, row in df.iterrows():
            times = [row['Q1_Time_Seconds'], row['Q2_Time_Seconds'], row['Q3_Time_Seconds']]
            valid_times = [t for t in times if pd.notna(t)]
            if valid_times:
                df.at[idx, 'Best_Time_Seconds'] = min(valid_times)
        
        # Sort by position
        df = df.sort_values('Position').reset_index(drop=True)
        
        print(f"Successfully loaded qualifying data for {year} {session_name} Grand Prix")
        print(f"Total drivers: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please check if the year and session name are correct.")
        print("Common session names: 'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami', 'Monaco', 'Spain', 'Canada', 'Austria', 'British', 'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'")
        return None