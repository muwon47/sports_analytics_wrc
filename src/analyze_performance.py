import pandas as pd
import numpy as np

def driver_ranking(df):
    """Compute overall driver ranking based on average stage position and time."""
    # Average stage position (lower is better)
    pos_rank = df.groupby('driver')['stage_position'].mean().sort_values()
    
    # Average time difference to winner (seconds per stage)
    time_rank = df.groupby('driver')['time_diff_to_winner_sec'].mean().sort_values()
    
    # Win count (number of stage wins)
    stage_wins = df[df['stage_position'] == 1].groupby('driver').size().sort_values(ascending=False)
    
    ranking = pd.DataFrame({
        'avg_stage_position': pos_rank,
        'avg_time_diff_to_winner_sec': time_rank,
        'stage_wins': stage_wins
    }).fillna(0)
    
    # Composite score: lower avg position + lower time diff + higher wins
    ranking['composite_score'] = (ranking['avg_stage_position'].rank(pct=True) +
                                  ranking['avg_time_diff_to_winner_sec'].rank(pct=True) -
                                  ranking['stage_wins'].rank(pct=True)) / 3
    ranking = ranking.sort_values('composite_score')
    return ranking

def consistency_analysis(df):
    """Compute consistency metrics per driver for the current dataframe slice."""
    grouped = df.groupby('driver')['stage_time_sec']
    consistency_df = grouped.agg(
        stages_completed='count',
        mean_stage_time_sec='mean',
        std_stage_time_sec=lambda values: values.std(ddof=0)
    )

    consistency_df['cv_stage_time'] = np.where(
        consistency_df['mean_stage_time_sec'].eq(0),
        0.0,
        consistency_df['std_stage_time_sec'] / consistency_df['mean_stage_time_sec']
    )

    numeric_cols = ['mean_stage_time_sec', 'std_stage_time_sec', 'cv_stage_time']
    consistency_df[numeric_cols] = consistency_df[numeric_cols].round(4)
    return consistency_df.sort_values(['cv_stage_time', 'std_stage_time_sec', 'driver'])

def car_performance(df):
    """Compare performance across different cars."""
    car_stats = df.groupby('car').agg({
        'stage_time_sec': 'mean',
        'avg_speed_kmh': 'mean',
        'stage_position': 'mean',
        'time_diff_to_winner_sec': 'mean'
    }).round(2)
    return car_stats.sort_values('stage_time_sec')

def rally_specific_analysis(df, rally_name=None):
    """Analyze performance for a specific rally."""
    if rally_name:
        df = df[df['rally'] == rally_name]
    rally_stats = df.groupby(['year', 'rally', 'driver'])['stage_position'].mean().unstack()
    return rally_stats
