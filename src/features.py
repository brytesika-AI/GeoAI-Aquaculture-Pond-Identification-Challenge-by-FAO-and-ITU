import re
import numpy as np
import pandas as pd

def parse_monthly_columns(df):
    """
    Dynamically scans columns to identify band prefixes and monthly suffixes.
    Supports B2/B02, B3/B03, B4/B04, B8/B08, B11, B12, VV, VH.
    Month suffixes can be 01-12, 1-12, or month names.
    Returns:
        months: list of sorted month identifiers (e.g. ['01', '02', ...])
        mappings: dict of month_id -> {band_name -> column_name}
    """
    # Regex to capture band prefix and month identifier
    # Band prefixes: B02, B2, B03, B3, B04, B4, B08, B8, B11, B12, VV, VH
    pattern = re.compile(r"^(B0?[2348]|B1[12]|VV|VH)[_-]?(0?[1-9]|1[0-2])$", re.IGNORECASE)
    
    mappings = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            band_raw, month_raw = match.groups()
            # Normalize band prefix
            band = band_raw.upper()
            if band in ['B2', 'B02']: band = 'BLUE'
            elif band in ['B3', 'B03']: band = 'GREEN'
            elif band in ['B4', 'B04']: band = 'RED'
            elif band in ['B8', 'B08']: band = 'NIR'
            elif band == 'B11': band = 'SWIR1'
            elif band == 'B12': band = 'SWIR2'
            
            # S1 polarizations
            elif band == 'VV': band = 'VV'
            elif band == 'VH': band = 'VH'
            
            # Normalize month identifier to integer/string
            month = f"{int(month_raw):02d}"
            
            if month not in mappings:
                mappings[month] = {}
            mappings[month][band] = col
            
    # Keep only months that have at least some bands
    sorted_months = sorted(list(mappings.keys()))
    return sorted_months, mappings

def compute_spectral_indices(df, months, mappings):
    """
    Computes spectral water and vegetation indices for each month:
    - NDVI = (NIR - RED) / (NIR + RED)
    - NDWI = (GREEN - NIR) / (GREEN + NIR)
    - MNDWI = (GREEN - SWIR1) / (GREEN + SWIR1)
    - AWEI = 4*(GREEN - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    - VH_VV_ratio = VH / VV
    - VV_VH_diff = VV - VH
    Returns the dataframe with calculated monthly features added.
    """
    df_feat = df.copy()
    eps = 1e-6
    
    for month in months:
        m_map = mappings[month]
        
        # S2 bands
        blue = df_feat[m_map['BLUE']] if 'BLUE' in m_map else None
        green = df_feat[m_map['GREEN']] if 'GREEN' in m_map else None
        red = df_feat[m_map['RED']] if 'RED' in m_map else None
        nir = df_feat[m_map['NIR']] if 'NIR' in m_map else None
        swir1 = df_feat[m_map['SWIR1']] if 'SWIR1' in m_map else None
        swir2 = df_feat[m_map['SWIR2']] if 'SWIR2' in m_map else None
        
        # S1 bands
        vv = df_feat[m_map['VV']] if 'VV' in m_map else None
        vh = df_feat[m_map['VH']] if 'VH' in m_map else None
        
        # 1. NDVI
        if nir is not None and red is not None:
            df_feat[f'NDVI_{month}'] = (nir - red) / (nir + red + eps)
            
        # 2. NDWI
        if green is not None and nir is not None:
            df_feat[f'NDWI_{month}'] = (green - nir) / (green + nir + eps)
            
        # 3. MNDWI
        if green is not None and swir1 is not None:
            df_feat[f'MNDWI_{month}'] = (green - swir1) / (green + swir1 + eps)
            
        # 4. AWEI (Automated Water Extraction Index)
        if green is not None and swir1 is not None and nir is not None and swir2 is not None:
            df_feat[f'AWEI_{month}'] = 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
            
        # 5. SAR features
        if vh is not None and vv is not None:
            df_feat[f'VH_VV_ratio_{month}'] = vh / (vv + eps)
            df_feat[f'VV_VH_diff_{month}'] = vv - vh
            
    return df_feat

def compute_temporal_aggregations(df, months, raw_bands, derived_indices):
    """
    Computes temporal aggregations across the 12 months for every raw band and index:
    - mean, std, min, max, range, median, IQR
    - linear trend slope (regression slope over 1 to 12)
    - seasonality (first harmonic amplitude of FFT)
    """
    df_agg = df.copy()
    num_months = len(months)
    month_indices = np.arange(1, num_months + 1)
    
    # Simple linear slope helper
    def get_slope(y_series):
        # y_series shape: (n_samples, n_months)
        # x: month_indices (1..12)
        # slope = Cov(x, y) / Var(x)
        x_mean = np.mean(month_indices)
        x_var = np.var(month_indices)
        cov = np.mean((month_indices - x_mean) * (y_series - np.mean(y_series, axis=1, keepdims=True)), axis=1)
        return cov / (x_var + 1e-8)
        
    # Seasonality helper (first harmonic amplitude of FFT)
    def get_seasonality(y_series):
        # Compute real FFT and take absolute value of the 1st harmonic (index 1)
        fft_vals = np.fft.rfft(y_series, axis=1)
        if fft_vals.shape[1] > 1:
            return np.abs(fft_vals[:, 1])
        return np.zeros(y_series.shape[0])

    features_to_agg = raw_bands + derived_indices
    for feat in features_to_agg:
        # Get all monthly columns for this feature
        monthly_cols = [f"{feat}_{m}" for m in months]
        # Check if all these columns exist in the DataFrame
        existing_cols = [c for c in monthly_cols if c in df_agg.columns]
        if len(existing_cols) < num_months:
            continue
            
        # Convert to numpy array of shape (n_samples, n_months)
        vals = df_agg[existing_cols].values
        
        # 1. Standard aggregations
        df_agg[f'{feat}_mean'] = np.mean(vals, axis=1)
        df_agg[f'{feat}_std'] = np.std(vals, axis=1)
        df_agg[f'{feat}_min'] = np.min(vals, axis=1)
        df_agg[f'{feat}_max'] = np.max(vals, axis=1)
        df_agg[f'{feat}_range'] = df_agg[f'{feat}_max'] - df_agg[f'{feat}_min']
        df_agg[f'{feat}_median'] = np.median(vals, axis=1)
        df_agg[f'{feat}_iqr'] = np.percentile(vals, 75, axis=1) - np.percentile(vals, 25, axis=1)
        
        # 2. Linear trend slope
        df_agg[f'{feat}_slope'] = get_slope(vals)
        
        # 3. Seasonality amplitude (FFT first harmonic)
        df_agg[f'{feat}_seasonality'] = get_seasonality(vals)
        
        # 4. Month of max and min
        df_agg[f'{feat}_month_max'] = np.argmax(vals, axis=1) + 1
        df_agg[f'{feat}_month_min'] = np.argmin(vals, axis=1) + 1
        
        # 5. Count of months above water threshold (for water indices)
        if 'NDWI' in feat or 'MNDWI' in feat:
            # Persistent water typically has positive NDWI/MNDWI
            df_agg[f'{feat}_water_months'] = np.sum(vals > 0.0, axis=1)
            
    return df_agg

def extract_all_features(df):
    """
    Main function to run the entire feature engineering pipeline.
    """
    months, mappings = parse_monthly_columns(df)
    if not months:
        raise ValueError("Could not find any valid monthly satellite data columns in the DataFrame.")
        
    print(f"Detected {len(months)} monthly composites: {months}")
    
    # Compute derived spectral indices per month
    df_feat = compute_spectral_indices(df, months, mappings)
    
    # Identify raw bands and derived indices
    raw_bands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'VV', 'VH']
    # Filter raw bands that are actually mapped
    mapped_raw_bands = []
    first_month_map = mappings[months[0]]
    for rb in raw_bands:
        if rb in first_month_map:
            mapped_raw_bands.append(rb)
            
    derived_indices = ['NDVI', 'NDWI', 'MNDWI', 'AWEI', 'VH_VV_ratio', 'VV_VH_diff']
    
    # Run temporal aggregation
    df_feat = compute_temporal_aggregations(df_feat, months, mapped_raw_bands, derived_indices)
    
    return df_feat
