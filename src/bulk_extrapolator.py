import os
import sys
import pandas as pd
from src.agents.agentic_extrapolator import AgenticGeospatialExtrapolator

def bulk_extrapolate(input_csv_path: str, output_csv_path: str, year: int = 2026):
    """
    Reads a CSV file containing latitude and longitude coordinates,
    fetches monthly S1/S2 satellite composites from Google Earth Engine,
    extracts features, runs the trained ensemble model, and writes predictions.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input coordinates file '{input_csv_path}' not found.")
        sys.exit(1)
        
    print(f"Loading coordinates from '{input_csv_path}'...")
    df_coords = pd.read_csv(input_csv_path)
    
    # Verify coordinates exist
    lat_col = [c for c in df_coords.columns if c.lower() in ['lat', 'latitude']][0]
    lon_col = [c for c in df_coords.columns if c.lower() in ['lon', 'longitude']][0]
    
    print(f"Using coordinate columns: '{lat_col}' and '{lon_col}'")
    print("Initializing Agentic Geospatial Extrapolator and loading models...")
    extrapolator = AgenticGeospatialExtrapolator()
    
    all_features = []
    probabilities = []
    
    total_rows = len(df_coords)
    print(f"Processing {total_rows} locations...")
    
    for idx, row in df_coords.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        print(f"\n[{idx+1}/{total_rows}] Processing point: {lat:.6f}, {lon:.6f}")
        
        # 1. Fetch monthly composites (GEE or Simulator)
        raw_data = extrapolator.fetch_gee_composites(lat, lon, year=year)
        
        # 2. Engineer monthly spectral indices and temporal stats
        df_feat = extrapolator.engineer_extrapolation_features(raw_data)
        all_features.append(df_feat)
        
        # 3. Run ensemble inference
        prob = extrapolator.run_ensemble_inference(df_feat)
        probabilities.append(prob)
        
    # Combine engineered features into one DataFrame
    df_features_all = pd.concat(all_features, ignore_index=True)
    
    # Construct output DataFrame
    df_output = df_coords.copy()
    df_output['aquaculture_probability'] = probabilities
    # 1 if probability >= 0.5 (or custom threshold)
    df_output['predicted_class'] = (df_output['aquaculture_probability'] >= 0.5).astype(int)
    
    # Save predictions
    df_output.to_csv(output_csv_path, index=False)
    print(f"\nSUCCESS: Bulk extrapolation completed. Predictions saved to '{output_csv_path}'.")
    print(df_output.head())

if __name__ == '__main__':
    # Usage Example:
    # Create a dummy coordinate list if none exists to demonstrate usage
    input_dummy_path = './data/africa_coordinates.csv'
    output_dummy_path = './outputs/africa_predictions.csv'
    
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    
    if not os.path.exists(input_dummy_path):
        print("Creating sample coordinates CSV for Africa...")
        # 1. Kajjansi Aquaculture Station, Uganda
        # 2. Lake Victoria Open Water (Uganda/Kenya/Tanzania border)
        # 3. Niger Delta wetlands, Nigeria
        # 4. Dry savannah coordinates
        sample_df = pd.DataFrame({
            'location_name': ['Kajjansi Fish Farm', 'Lake Victoria Open Water', 'Niger Delta Wetland', 'Savanna Land'],
            'latitude': [0.200000, -1.500000, 5.200000, 9.500000],
            'longitude': [32.540000, 32.800000, 6.200000, 20.100000]
        })
        sample_df.to_csv(input_dummy_path, index=False)
        
    bulk_extrapolate(input_dummy_path, output_dummy_path)
