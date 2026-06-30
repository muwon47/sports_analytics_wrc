import pandas as pd
import numpy as np
import os

def generate_synthetic_data(output_path='data/wrc_data.csv'):
    """Generate synthetic WRC dataset if not already present."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}. Loading...")
        return pd.read_csv(output_path)
    
    print("Generating synthetic WRC dataset...")
    
    np.random.seed(42)
    
    # Parameters
    rallies = ['Monte Carlo', 'Sweden', 'Portugal', 'Finland', 'Wales GB']
    years = [2021, 2022, 2023]
    drivers = ['Rovanpera', 'Evans', 'Ogier', 'Tanak', 'Neuville', 'Breen', 'Lappi']
    co_drivers = ['Halttunen', 'Martin', 'Landais', 'Jarveoja', 'Wydaeghe', 'Fulton', 'Ferm']
    cars = ['Toyota Yaris', 'Hyundai i20', 'Ford Puma']
    
    records = []
    
    for year in years:
        for rally in rallies:
            # Number of stages per rally (between 15 and 22)
            n_stages = np.random.randint(15, 23)
            stage_lengths = np.random.uniform(8, 45, n_stages).round(1)  # km
            
            for stage_idx in range(n_stages):
                stage_name = f"SS{stage_idx+1}"
                length = stage_lengths[stage_idx]
                
                for driver, co_driver in zip(drivers, co_drivers):
                    # Base performance: driver skill factor (lower is better)
                    skill = {'Rovanpera': 0.92, 'Evans': 0.96, 'Ogier': 0.94,
                             'Tanak': 0.95, 'Neuville': 0.97, 'Breen': 1.00,
                             'Lappi': 0.99}[driver]
                    
                    # Car effect
                    car = np.random.choice(cars)
                    car_factor = {'Toyota Yaris': 0.98, 'Hyundai i20': 1.00, 'Ford Puma': 1.02}[car]
                    
                    # Stage-specific variation (weather, surface)
                    stage_variation = np.random.normal(1.0, 0.05)
                    
                    # Expected speed (km/h) - realistic rally speeds 80-120 km/h
                    expected_speed = 100 * skill * car_factor * stage_variation
                    expected_speed = np.clip(expected_speed, 70, 140)
                    
                    # Stage time (seconds) = distance / speed * 3600
                    stage_time = (length / expected_speed) * 3600
                    
                    # Add random error (driver inconsistency)
                    stage_time += np.random.normal(0, stage_time * 0.02)
                    
                    # Position (1 to number of drivers) - lower time = better position
                    # We'll sort later, but for generation we keep raw time
                    
                    records.append({
                        'year': year,
                        'rally': rally,
                        'stage_id': stage_idx + 1,
                        'stage_name': stage_name,
                        'stage_length_km': length,
                        'driver': driver,
                        'co_driver': co_driver,
                        'car': car,
                        'stage_time_sec': round(stage_time, 2)
                    })
    
    df = pd.DataFrame(records)
    
    # Compute stage position per rally-stage
    df['stage_position'] = df.groupby(['year', 'rally', 'stage_id'])['stage_time_sec'] \
                             .rank(method='dense').astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")
    return df

def load_data(filepath='data/wrc_data.csv'):
    """Load existing CSV or generate synthetic data."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return generate_synthetic_data(filepath)