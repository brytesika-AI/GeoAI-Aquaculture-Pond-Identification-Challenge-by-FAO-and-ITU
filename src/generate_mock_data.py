import os
import numpy as np
import pandas as pd

def generate_mock_datasets():
    print("Generating mock datasets for pipeline validation...")
    os.makedirs('./data', exist_ok=True)
    
    np.random.seed(42)
    
    # 1. Define columns
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'VV', 'VH']
    months = [str(i) for i in range(1, 13)]
    
    cols = ['id', 'latitude', 'longitude']
    for m in months:
        for b in bands:
            cols.append(f"{b}_{m}")
            
    # Number of rows
    n_train = 953
    n_test = 858
    
    # Generate coordinates centered around Mekong Delta (Vietnam)
    train_lat = np.random.uniform(9.5, 10.5, n_train)
    train_lon = np.random.uniform(105.0, 106.5, n_train)
    test_lat = np.random.uniform(9.5, 10.5, n_test)
    test_lon = np.random.uniform(105.0, 106.5, n_test)
    
    # Generate Train data
    train_data = {
        'id': [f"TRAIN_{i:04d}" for i in range(n_train)],
        'latitude': train_lat,
        'longitude': train_lon
    }
    
    # Generate Test data
    test_data = {
        'id': [f"TEST_{i:04d}" for i in range(n_test)],
        'latitude': test_lat,
        'longitude': test_lon
    }
    
    # Simulate realistic physical ranges for bands
    # Water has low reflectance in NIR (B8), high MNDWI (Green-SWIR), low SAR backscatter
    # We will make some points represent ponds (target=1) and others represent land (target=0)
    train_target = np.random.binomial(1, 0.25, n_train)
    train_data['target'] = train_target
    
    def fill_mock_satellite_data(data_dict, n_rows, targets=None):
        for m in months:
            for b in bands:
                col_name = f"{b}_{m}"
                # If target is pond (1), make it look like water:
                # low NIR (B8) (~0.05), high GREEN (B3) (~0.15), low RED (B4) (~0.08), low SAR backscatter (~-18.0)
                if targets is not None:
                    water_vals = np.random.normal(0.08, 0.02, n_rows) # default values
                    for idx in range(n_rows):
                        is_pond = targets[idx] == 1
                        if is_pond:
                            if b == 'B8': val = np.random.normal(0.04, 0.01)
                            elif b == 'B3': val = np.random.normal(0.18, 0.03)
                            elif b == 'B4': val = np.random.normal(0.06, 0.01)
                            elif b == 'B11': val = np.random.normal(0.03, 0.01)
                            elif b == 'B12': val = np.random.normal(0.02, 0.01)
                            elif b == 'B2': val = np.random.normal(0.12, 0.02)
                            elif b == 'VV': val = np.random.normal(-18.0, 2.0)
                            elif b == 'VH': val = np.random.normal(-24.0, 2.0)
                        else:
                            # Land: high NIR (B8) (~0.4), lower GREEN, higher SWIR, higher SAR
                            if b == 'B8': val = np.random.normal(0.35, 0.05)
                            elif b == 'B3': val = np.random.normal(0.10, 0.02)
                            elif b == 'B4': val = np.random.normal(0.08, 0.02)
                            elif b == 'B11': val = np.random.normal(0.20, 0.04)
                            elif b == 'B12': val = np.random.normal(0.15, 0.03)
                            elif b == 'B2': val = np.random.normal(0.05, 0.01)
                            elif b == 'VV': val = np.random.normal(-8.0, 1.5)
                            elif b == 'VH': val = np.random.normal(-14.0, 1.5)
                        water_vals[idx] = val
                    data_dict[col_name] = water_vals
                else:
                    # Test set: mix of water and land
                    test_targets = np.random.binomial(1, 0.25, n_rows)
                    water_vals = np.random.normal(0.08, 0.02, n_rows)
                    for idx in range(n_rows):
                        is_pond = test_targets[idx] == 1
                        if is_pond:
                            if b == 'B8': val = np.random.normal(0.04, 0.01)
                            elif b == 'B3': val = np.random.normal(0.18, 0.03)
                            elif b == 'B4': val = np.random.normal(0.06, 0.01)
                            elif b == 'B11': val = np.random.normal(0.03, 0.01)
                            elif b == 'B12': val = np.random.normal(0.02, 0.01)
                            elif b == 'B2': val = np.random.normal(0.12, 0.02)
                            elif b == 'VV': val = np.random.normal(-18.0, 2.0)
                            elif b == 'VH': val = np.random.normal(-24.0, 2.0)
                        else:
                            if b == 'B8': val = np.random.normal(0.35, 0.05)
                            elif b == 'B3': val = np.random.normal(0.10, 0.02)
                            elif b == 'B4': val = np.random.normal(0.08, 0.02)
                            elif b == 'B11': val = np.random.normal(0.20, 0.04)
                            elif b == 'B12': val = np.random.normal(0.15, 0.03)
                            elif b == 'B2': val = np.random.normal(0.05, 0.01)
                            elif b == 'VV': val = np.random.normal(-8.0, 1.5)
                            elif b == 'VH': val = np.random.normal(-14.0, 1.5)
                        water_vals[idx] = val
                    data_dict[col_name] = water_vals

    fill_mock_satellite_data(train_data, n_train, train_target)
    fill_mock_satellite_data(test_data, n_test, None)
    
    # Save DataFrames
    pd.DataFrame(train_data).to_csv('./data/Train.csv', index=False)
    pd.DataFrame(test_data).to_csv('./data/Test.csv', index=False)
    
    # Create SampleSubmission.csv
    sample_sub = pd.DataFrame({
        'id': test_data['id'],
        'target': np.random.uniform(0.0, 1.0, n_test) # placeholder probabilities
    })
    sample_sub.to_csv('./data/SampleSubmission.csv', index=False)
    
    # Create a dummy trustworthiness pdf
    with open('./data/Trustworthiness_Evaluation.pdf', 'w') as f:
        f.write("%PDF-1.4 mock pdf document for testing")
        
    print("SUCCESS: Generated Train.csv, Test.csv, SampleSubmission.csv, and Trustworthiness_Evaluation.pdf in ./data/.")

if __name__ == '__main__':
    generate_mock_datasets()
