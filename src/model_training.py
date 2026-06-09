import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import gc
import os
import h5py

def train_metr_la_model(data_path='metr-la.h5', model_save_path='traffic_xgb_model.json'):
    print("[Advanced ML Pipeline] Initializing Feature Engineering for METR-LA...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found. Please ensure it is in the root directory.")
        
    # --- FIX: Robustly find the 2D data array inside the HDF5 structure ---
    print("[ML Engine] Loading raw HDF5 matrix via h5py...")
    with h5py.File(data_path, 'r') as f:
        def find_2d_dataset(group):
            for k in group.keys():
                item = group[k]
                # Look for a 2D dataset (the actual speed matrix)
                if isinstance(item, h5py.Dataset) and item.ndim == 2:
                    return item[:]
                elif isinstance(item, h5py.Group):
                    res = find_2d_dataset(item)
                    if res is not None:
                        return res
            return None
            
        raw_data = find_2d_dataset(f)
        if raw_data is None:
            raise ValueError("Could not find the 2D speed matrix inside the HDF5 file.")

    # Convert to Pandas DataFrame (Standard METR-LA format: Rows=Time, Cols=Sensors)
    df_speed = pd.DataFrame(raw_data)
    num_timestamps, num_sensors = df_speed.shape
    print(f"-> Loaded matrix shape: {num_timestamps} timestamps x {num_sensors} sensors.")

    # Generate a standard 5-minute interval DatetimeIndex (METR-LA standard)
    timestamps = pd.date_range(start='2012-03-01', periods=num_timestamps, freq='5min')
    df_speed.index = timestamps

    hour_of_day = (timestamps.hour + timestamps.minute / 60.0).values
    day_of_week = timestamps.dayofweek.values
    sensor_ids = np.arange(num_sensors)

    hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)

    np.random.seed(42)
    sample_size = 200000
    rand_timestamps = np.random.randint(12, num_timestamps, size=sample_size)
    rand_sensors = np.random.randint(0, num_sensors, size=sample_size)

    X_list, y_list = [], []

    print("[ML Engine] Engineering lag matrices [Hour_Sin, Hour_Cos, Day, Sensor, Lag_15m, Lag_1h]...")
    for i in range(sample_size):
        t_idx, s_idx = rand_timestamps[i], rand_sensors[i]
        speed_val = df_speed.iloc[t_idx, s_idx]
        
        if speed_val > 5.0:
            lag_15m = df_speed.iloc[t_idx - 3, s_idx]
            lag_1h = df_speed.iloc[t_idx - 12, s_idx]
            
            if lag_15m <= 5.0: lag_15m = 55.0
            if lag_1h <= 5.0: lag_1h = 55.0
            
            X_list.append([hour_sin[t_idx], hour_cos[t_idx], day_of_week[t_idx], sensor_ids[s_idx], lag_15m, lag_1h])
            y_list.append(speed_val)

    X, y = np.array(X_list), np.array(y_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print("[ML Engine] Fine-tuning Deep-Tree XGBoost Production Estimator...")
    traffic_model = XGBRegressor(
        n_estimators=120, max_depth=8, learning_rate=0.12, 
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    traffic_model.fit(X_train, y_train)
    traffic_model.save_model(model_save_path)
    
    r2_score = traffic_model.score(X_test, y_test)
    print(f"-> Successfully saved serialized production weights to: {model_save_path}")
    print(f"-> Model Validation R² Score: {r2_score:.4f}")

    del df_speed
    gc.collect()
    return traffic_model, r2_score