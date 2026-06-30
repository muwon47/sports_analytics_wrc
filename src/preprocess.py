import pandas as pd
import numpy as np

def clean_data(df):
    """Basic cleaning: remove duplicates, handle missing values."""
    df = df.drop_duplicates()
    # No missing values in synthetic data, but in real data you might have:
    for col in ['stage_time_sec', 'stage_length_km']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

def engineer_features(df):
    """Add new columns: average speed, performance score, etc."""
    df = df.copy()
    # Average speed (km/h)
    df['avg_speed_kmh'] = (df['stage_length_km'] / df['stage_time_sec']) * 3600
    
    # Performance relative to stage winner (lower = better)
    # First, get winner time per stage
    winner_time = df.groupby(['year', 'rally', 'stage_id'])['stage_time_sec'].transform('min')
    df['time_diff_to_winner_sec'] = df['stage_time_sec'] - winner_time
    
    # Normalized performance score (0 = winner, 1 = worst in stage)
    max_diff = df.groupby(['year', 'rally', 'stage_id'])['time_diff_to_winner_sec'].transform('max')
    df['performance_score'] = df['time_diff_to_winner_sec'] / max_diff.replace(0, np.nan)
    df['performance_score'] = df['performance_score'].fillna(0)  # winner gets 0
    
    # Consistency: rolling average of last 3 stage times per driver (within same rally)
    df = df.sort_values(['year', 'rally', 'driver', 'stage_id'])
    df['prev_3_stage_avg_sec'] = df.groupby(['year', 'rally', 'driver'])['stage_time_sec'] \
                                   .transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    return df

def get_summary_stats(df):
    """Return summary statistics for the dataset."""
    return df.describe()