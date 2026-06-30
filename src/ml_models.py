import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.analyze_performance import consistency_analysis

def perform_kmeans_clustering(df, n_clusters=3):
    """
    Cluster drivers based on their average stage time and consistency (CV).
    """
    # Get driver-level metrics
    driver_stats = consistency_analysis(df)
    
    # We need drivers with enough stages to have a valid CV
    valid_drivers = driver_stats[driver_stats['stages_completed'] >= 2].copy().reset_index()
    
    if len(valid_drivers) < n_clusters:
        return valid_drivers
        
    features = ['mean_stage_time_sec', 'cv_stage_time']
    X = valid_drivers[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    valid_drivers['cluster'] = kmeans.fit_predict(X_scaled)
    
    return valid_drivers

def train_random_forest(df):
    """
    Train a Random Forest to predict stage_time_sec based on stage_length_km, car, and driver.
    Returns the trained model, feature importances, and a sample dataframe.
    """
    df_rf = df.dropna(subset=['stage_length_km', 'car', 'driver', 'stage_time_sec']).copy()
    
    if len(df_rf) == 0:
        return None, None, None
        
    # One-hot encode categorical features to avoid ordinal bias
    df_rf = pd.get_dummies(df_rf, columns=['car', 'driver'], prefix=['car', 'driver'], drop_first=False)
    
    feature_columns = ['stage_length_km'] + [col for col in df_rf.columns if col.startswith('car_') or col.startswith('driver_')]
    X = df_rf[feature_columns]
    y = df_rf['stage_time_sec']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    raw_importances = pd.Series(rf.feature_importances_, index=feature_columns)
    importance_summary = pd.DataFrame({
        'feature': [
            'Stage Length (km)',
            'Car Model',
            'Driver'
        ],
        'importance': [
            raw_importances['stage_length_km'],
            raw_importances[[c for c in raw_importances.index if c.startswith('car_')]].sum(),
            raw_importances[[c for c in raw_importances.index if c.startswith('driver_')]].sum()
        ]
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return rf, importance_summary, df_rf

def detect_anomalies(df):
    """
    Use Isolation Forest to detect anomalous stage times (e.g., crashes or extremely fast runs).
    """
    df_anomaly = df.dropna(subset=['stage_length_km', 'avg_speed_kmh', 'stage_time_sec']).copy()
    
    if len(df_anomaly) < 10:
        df_anomaly['anomaly'] = 1 # 1 is normal, -1 is anomaly
        return df_anomaly
        
    features = ['stage_length_km', 'avg_speed_kmh', 'stage_time_sec']
    X = df_anomaly[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # contamination is the proportion of outliers in the dataset
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    
    # Predict returns 1 for inliers, -1 for outliers
    df_anomaly['anomaly'] = iso_forest.fit_predict(X_scaled)
    
    return df_anomaly
