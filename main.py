import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.load_data import load_data
from src.preprocess import clean_data, engineer_features, get_summary_stats
from src.analyze_performance import driver_ranking, consistency_analysis, car_performance, rally_specific_analysis
from src.visualize import (plot_driver_ranking, plot_avg_stage_time_by_driver, 
                           plot_performance_over_stages, plot_car_comparison,
                           plot_time_trend, plot_heatmap_performance)
from src.ml_models import perform_kmeans_clustering, train_random_forest, detect_anomalies

def main():
    print("=== WRC Sports Analytics Project ===")
    
    # 1. Load data
    df = load_data()
    print(f"Loaded {len(df)} records.")
    
    # 2. Preprocess
    df = clean_data(df)
    df = engineer_features(df)
    
    # 3. Basic summary
    print("\n--- Summary Statistics ---")
    print(get_summary_stats(df))
    
    # 4. Performance analysis
    print("\n--- Driver Ranking ---")
    ranking = driver_ranking(df)
    print(ranking.head())
    
    print("\n--- Consistency Analysis (lowest CV = most consistent) ---")
    consistency = consistency_analysis(df)
    print(consistency.head())
    
    print("\n--- Car Performance Comparison ---")
    car_stats = car_performance(df)
    print(car_stats)
    
    # 5. Visualizations
    print("\n--- Generating Plots ---")
    plot_driver_ranking(ranking)
    plot_avg_stage_time_by_driver(df)
    plot_performance_over_stages(df)
    plot_car_comparison(car_stats)
    plot_time_trend(df)
    plot_heatmap_performance(df)
    
    # 6. Extra: Rally-specific analysis example
    rally_name = df['rally'].iloc[0]  # first rally in dataset
    rally_stats = rally_specific_analysis(df, rally_name)
    print(f"\n--- Average stage position in {rally_name} (rows=driver, cols=year) ---")
    print(rally_stats)
    
    # 7. AI/ML Analytics
    print("\n--- AI/ML Analytics ---")
    print("1. K-Means Clustering (Drivers):")
    clusters_df = perform_kmeans_clustering(df, n_clusters=3)
    if not clusters_df.empty:
        print(clusters_df[['driver', 'cluster']].head(10))
    
    print("\n2. Random Forest Feature Importances:")
    rf_model, importances, rf_data = train_random_forest(df)
    if importances is not None:
        print(importances)
        
    print("\n3. Isolation Forest Anomalies:")
    anomalies_df = detect_anomalies(df)
    if not anomalies_df.empty:
        anomaly_count = (anomalies_df['anomaly'] == -1).sum()
        print(f"Detected {anomaly_count} anomalies out of {len(anomalies_df)} valid stage records.")
    
    print("\nProject execution completed.")

if __name__ == "__main__":
    main()