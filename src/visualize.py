import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_style():
    """Set consistent styling for all plots with a high-tech dark theme."""
    sns.set_theme(style='darkgrid', rc={
        'axes.facecolor': '#15192E',
        'figure.facecolor': '#0A0D1A',
        'grid.color': '#2A2E45',
        'text.color': '#E2E8F0',
        'axes.labelcolor': '#E2E8F0',
        'xtick.color': '#A0AEC0',
        'ytick.color': '#A0AEC0',
        'font.family': 'sans-serif'
    })
    plt.rcParams['text.color'] = '#E2E8F0'
    plt.rcParams['axes.labelcolor'] = '#E2E8F0'
    plt.rcParams['xtick.color'] = '#A0AEC0'
    plt.rcParams['ytick.color'] = '#A0AEC0'
    plt.rcParams['figure.facecolor'] = '#0A0D1A'
    plt.rcParams['axes.facecolor'] = '#15192E'
    plt.rcParams['figure.figsize'] = (10, 6)

def plot_driver_ranking(ranking_df):
    """
    Bar chart of driver composite scores.
    Returns matplotlib figure.
    """
    set_style()
    fig, ax = plt.subplots()
    colors = sns.color_palette("flare_r", len(ranking_df))
    ranking_df['composite_score'].sort_values().plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Driver Composite Performance Score (lower is better)')
    ax.set_xlabel('Composite Score')
    plt.tight_layout()
    return fig

def plot_avg_stage_time_by_driver(df):
    """
    Boxplot of stage times per driver.
    Returns matplotlib figure.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x='driver',
        y='stage_time_sec',
        hue='driver',
        dodge=False,
        palette='crest',
        ax=ax
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title('Distribution of Stage Times per Driver')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Stage Time (seconds)')
    plt.tight_layout()
    return fig

def plot_kmeans_clusters(df_clusters):
    """
    Scatter plot of K-Means clusters (Mean Stage Time vs CV Stage Time).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'cluster' not in df_clusters.columns:
        ax.text(0.5, 0.5, "No cluster data available", ha='center', va='center', transform=ax.transAxes)
        return fig
        
    sns.scatterplot(
        data=df_clusters, 
        x='mean_stage_time_sec', 
        y='cv_stage_time', 
        hue='cluster', 
        palette='cool',
        s=150,
        ax=ax
    )
    
    # Annotate points with driver names
    for idx, row in df_clusters.iterrows():
        ax.annotate(row['driver'], 
                    (row['mean_stage_time_sec'], row['cv_stage_time']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
                    
    ax.set_title('Driver Clusters based on Average Time & Consistency')
    ax.set_xlabel('Mean Stage Time (seconds)')
    ax.set_ylabel('Consistency (CV of Stage Time)')
    plt.tight_layout()
    return fig

def plot_feature_importance(importances_df):
    """
    Bar chart of Random Forest feature importances.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if importances_df is None or importances_df.empty:
        ax.text(0.5, 0.5, "No importance data available", ha='center', va='center', transform=ax.transAxes)
        return fig
        
    importances_df = importances_df.sort_values('importance', ascending=True)
    sns.barplot(data=importances_df, x='importance', y='feature', palette='flare', ax=ax)
    ax.set_title('Random Forest Feature Importance')
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('Feature')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=4)
    plt.tight_layout()
    return fig

def plot_anomalies(df_anomalies):
    """
    Scatter plot highlighting anomalies detected by Isolation Forest.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'anomaly' not in df_anomalies.columns:
        ax.text(0.5, 0.5, "No anomaly data available", ha='center', va='center', transform=ax.transAxes)
        return fig
        
    # Anomaly labels: 1 = normal, -1 = anomaly
    normal = df_anomalies[df_anomalies['anomaly'] == 1]
    anomalies = df_anomalies[df_anomalies['anomaly'] == -1]
    
    ax.scatter(normal['stage_length_km'], normal['stage_time_sec'], 
               color='#00F0FF', label='Normal', alpha=0.6, s=40)
    ax.scatter(anomalies['stage_length_km'], anomalies['stage_time_sec'], 
               color='#FF3366', label='Anomaly', alpha=0.9, s=100, marker='x', linewidths=2)
               
    ax.set_title('Anomaly Detection in Stage Times (Isolation Forest)')
    ax.set_xlabel('Stage Length (km)')
    ax.set_ylabel('Stage Time (seconds)')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_performance_over_stages(df, driver_list=None):
    """
    Line plot of stage positions over stage sequence for selected drivers.
    Returns matplotlib figure.
    """
    set_style()
    if driver_list is None:
        driver_list = df['driver'].unique()[:4]  # top 4 drivers by default
    df_filtered = df[df['driver'].isin(driver_list)]
    # Use a specific rally for clarity, e.g., latest year, first rally
    if not df_filtered.empty:
        sample_rally = df_filtered.groupby(['year', 'rally']).size().idxmax()
        df_rally = df_filtered[(df_filtered['year'] == sample_rally[0]) & 
                               (df_filtered['rally'] == sample_rally[1])]
        if not df_rally.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=df_rally, x='stage_id', y='stage_position', 
                         hue='driver', marker='o', linewidth=2.5, palette='flare', ax=ax)
            ax.set_title(f'Stage Position per Stage - {sample_rally[1]} {sample_rally[0]}')
            ax.set_xlabel('Stage Number')
            ax.set_ylabel('Stage Position')
            ax.invert_yaxis()  # lower position is better
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            return fig
    # Fallback: empty figure
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "No data available for the selected filters", 
            ha='center', va='center', transform=ax.transAxes)
    return fig

def plot_car_comparison(car_stats):
    set_style()
    fig, ax = plt.subplots()
    # The correct column name is 'stage_time_sec' (average per car)
    # If you're unsure, uncomment the next line to see all columns:
    # print(car_stats.columns)
    colors = sns.color_palette("crest_r", len(car_stats))
    car_stats['stage_time_sec'].sort_values().plot(kind='bar', color=colors, ax=ax)
    ax.set_title('Average Stage Time by Car Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    return fig

def plot_time_trend(df):
    """
    Trend of average stage times over years.
    Returns matplotlib figure.
    """
    set_style()
    fig, ax = plt.subplots()
    yearly_avg = df.groupby('year')['stage_time_sec'].mean().reset_index()
    ax.plot(yearly_avg['year'], yearly_avg['stage_time_sec'], marker='o', color='#00F0FF', linestyle='--', linewidth=2.5, markersize=8)
    ax.set_title('Average Stage Time Trend Across Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Stage Time (seconds)')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_heatmap_performance(df):
    """
    Heatmap of average performance score per driver and rally.
    Returns matplotlib figure.
    """
    set_style()
    pivot = df.pivot_table(index='driver', columns='rally', values='performance_score', aggfunc='mean')
    if pivot.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for heatmap", ha='center', va='center', transform=ax.transAxes)
        return fig
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='RdYlGn_r', center=0.5, linewidths=1.5, linecolor='#15192E', ax=ax)
    ax.set_title('Average Performance Score per Driver and Rally (lower is better)')
    plt.tight_layout()
    return fig
