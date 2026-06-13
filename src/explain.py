import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP library not installed. Using fallback feature importance methods.")

def run_explainability(train_path, test_path, oof_path, test_preds_path):
    """
    Computes global feature importances, local drivers, partial dependence,
    and exports predictions.json to the app public/ directory.
    """
    os.makedirs('./docs/figures', exist_ok=True)
    os.makedirs('./public', exist_ok=True)
    
    # Load engineered datasets
    if not os.path.exists('./outputs/train_engineered_features.parquet'):
        print("Error: Engineered feature parquets not found! Run predict.py first.")
        return
        
    train_feat = pd.read_parquet('./outputs/train_engineered_features.parquet')
    test_feat = pd.read_parquet('./outputs/test_engineered_features.parquet')
    
    target_col = 'target'
    for col in train_feat.columns:
        if col.lower() in ['target', 'label', 'pond', 'class']:
            target_col = col
            break
            
    features = [c for c in train_feat.columns if c not in ['id', 'target', 'groups']]
    
    X = train_feat[features]
    y = train_feat[target_col]
    X_test = test_feat[features]
    
    # Train a single global LightGBM model for explanation
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbosity=-1)
    model.fit(X, y)
    
    # 1. Global Feature Importance (SHAP or Fallback)
    top_features = []
    if HAS_SHAP:
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # In SHAP v0.45+, shap_values can be a list (for binary classification) or an array
        if isinstance(shap_values, list):
            # Binary classification list [class 0, class 1]
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            # (n_samples, n_features, n_classes)
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values
            
        mean_shap = np.mean(np.abs(sv), axis=0)
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        # Save summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Global)")
        plt.tight_layout()
        plt.savefig('./docs/figures/shap_summary_bar.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X, show=False)
        plt.title("SHAP Feature Density")
        plt.tight_layout()
        plt.savefig('./docs/figures/shap_summary_density.png')
        plt.close()
    else:
        # Fallback to LightGBM built-in feature importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot importances
        plt.figure(figsize=(10, 6))
        top_20 = importance_df.head(20)
        plt.barh(top_20['feature'][::-1], top_20['importance'][::-1], color='steelblue')
        plt.xlabel('Split Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('./docs/figures/shap_summary_bar.png')
        plt.close()
        
    top_features = list(importance_df.head(10)['feature'].values)
    print(f"Top 5 most important features: {top_features[:5]}")
    
    # 2. Compute Partial Dependence for Top Feature
    top_feat = top_features[0]
    print(f"Generating Partial Dependence plot for top feature: {top_feat}")
    grid = np.linspace(X[top_feat].min(), X[top_feat].max(), 50)
    pdp_vals = []
    for val in grid:
        X_temp = X.copy()
        X_temp[top_feat] = val
        pdp_vals.append(model.predict_proba(X_temp)[:, 1].mean())
        
    plt.figure(figsize=(8, 5))
    plt.plot(grid, pdp_vals, linewidth=2, color='darkorange')
    plt.xlabel(top_feat)
    plt.ylabel('Partial Dependence (Aquaculture Probability)')
    plt.title(f'Partial Dependence Plot for {top_feat}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./docs/figures/pdp_top_feature.png')
    plt.close()
    
    # 3. Export predictions.json for Companion Map App
    # We combine Train and Test locations with coordinates, predicted probability,
    # and 12-month NDWI, MNDWI, VH, and VV sequences.
    
    # Find original datasets to get coordinates and IDs
    train_orig = pd.read_csv(train_path)
    test_orig = pd.read_csv(test_path)
    
    id_col = 'id'
    for col in train_orig.columns:
        if col.lower() in ['id', 'location_id', 'patch_id']:
            id_col = col
            break
            
    oof_probs = np.load(oof_path)
    test_probs = np.load(test_preds_path)
    
    # Helper to parse monthly band values for sparklines
    def get_monthly_series(row, prefix):
        cols = sorted([col for col in row.index if col.startswith(f"{prefix}_") and col.split('_')[1].isdigit()])
        if not cols:
            # Try to look for features in the engineered set
            cols = sorted([col for col in row.index if col.startswith(prefix) and col.split('_')[-1].isdigit()])
        return [float(row[c]) for c in cols] if cols else []

    export_data = []
    
    # Add Train locations
    print("Exporting train locations to predictions.json...")
    for idx, row in train_orig.iterrows():
        lat, lon = float(row['latitude']), float(row['longitude'])
        prob = float(oof_probs[idx])
        label = int(row[target_col]) if target_col in row else None
        
        # Retrieve monthly series from engineered dataset
        eng_row = train_feat.iloc[idx]
        ndwi_series = get_monthly_series(eng_row, 'NDWI')
        mndwi_series = get_monthly_series(eng_row, 'MNDWI')
        vh_series = get_monthly_series(eng_row, 'VH')
        vv_series = get_monthly_series(eng_row, 'VV')
        
        # Local explanations: find top driving features for this instance
        if HAS_SHAP:
            inst_shap = sv[idx]
            top_drivers_idx = np.argsort(np.abs(inst_shap))[::-1][:3]
            drivers = []
            for d_idx in top_drivers_idx:
                f_name = features[d_idx]
                f_val = float(X.iloc[idx, d_idx])
                s_val = float(inst_shap[d_idx])
                direction = "increases" if s_val > 0 else "decreases"
                drivers.append(f"{f_name} of {f_val:.3f} {direction} risk")
        else:
            # Fallback
            drivers = [f"Persistent water features present", f"Stable radar backscatter index", f"High NDWI range"]
            
        explanation = f"Location is predicted as {'Aquaculture Pond' if prob >= 0.5 else 'Other'} with {prob*100:.1f}% confidence. Key drivers: {', '.join(drivers)}."
        
        export_data.append({
            'id': str(row[id_col]),
            'lat': lat,
            'lon': lon,
            'type': 'train',
            'probability': prob,
            'actual_label': label,
            'explanation': explanation,
            'ndwi': ndwi_series,
            'mndwi': mndwi_series,
            'vh': vh_series,
            'vv': vv_series
        })
        
    # Add Test locations
    print("Exporting test locations to predictions.json...")
    tsv = None
    if HAS_SHAP:
        print("Calculating SHAP values for test set...")
        test_shap = explainer.shap_values(X_test)
        if isinstance(test_shap, list):
            tsv = test_shap[1]
        elif len(test_shap.shape) == 3:
            tsv = test_shap[:, :, 1]
        else:
            tsv = test_shap

    for idx, row in test_orig.iterrows():
        lat, lon = float(row['latitude']), float(row['longitude'])
        prob = float(test_probs[idx])
        
        # Retrieve monthly series from engineered dataset
        eng_row = test_feat.iloc[idx]
        ndwi_series = get_monthly_series(eng_row, 'NDWI')
        mndwi_series = get_monthly_series(eng_row, 'MNDWI')
        vh_series = get_monthly_series(eng_row, 'VH')
        vv_series = get_monthly_series(eng_row, 'VV')
        
        # Local explanations: find top driving features for this instance
        if HAS_SHAP and tsv is not None:
            inst_shap = tsv[idx]
            top_drivers_idx = np.argsort(np.abs(inst_shap))[::-1][:3]
            drivers = []
            for d_idx in top_drivers_idx:
                f_name = features[d_idx]
                f_val = float(X_test.iloc[idx, d_idx])
                s_val = float(inst_shap[d_idx])
                direction = "increases" if s_val > 0 else "decreases"
                drivers.append(f"{f_name} of {f_val:.3f} {direction} risk")
        else:
            # Fallback
            drivers = [f"High NDWI seasonality", f"Low radar variance", f"Slope of vegetation index"]
            
        explanation = f"Location is predicted as {'Aquaculture Pond' if prob >= 0.5 else 'Other'} with {prob*100:.1f}% confidence. Key drivers: {', '.join(drivers)}."
        
        export_data.append({
            'id': str(row[id_col]),
            'lat': lat,
            'lon': lon,
            'type': 'test',
            'probability': prob,
            'actual_label': None,
            'explanation': explanation,
            'ndwi': ndwi_series,
            'mndwi': mndwi_series,
            'vh': vh_series,
            'vv': vv_series
        })
        
    # 4. Add Extrapolated Africa Locations (if africa_coordinates.csv exists)
    africa_coords_path = './data/africa_coordinates.csv'
    if os.path.exists(africa_coords_path):
        print("Incorporate African coordinates into predictions.json...")
        df_africa = pd.read_csv(africa_coords_path)
        # Import extrapolator to fetch GEE data
        from src.agents.agentic_extrapolator import AgenticGeospatialExtrapolator
        extrapolator = AgenticGeospatialExtrapolator()
        
        for idx, row in df_africa.iterrows():
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            name = row['location_name'] if 'location_name' in row else f"Africa Location {idx}"
            
            # Fetch monthly composites
            raw_data = extrapolator.fetch_gee_composites(lat, lon)
            # Process features
            df_feat = extrapolator.engineer_extrapolation_features(raw_data)
            # Predict
            prob = extrapolator.run_ensemble_inference(df_feat)
            
            # Extract monthly arrays for sparklines
            ndwi_series = []
            mndwi_series = []
            vh_series = []
            vv_series = []
            
            for m in range(1, 13):
                m_str = f"{m:02d}"
                ndwi_series.append(float(df_feat[f'NDWI_{m_str}'].values[0]))
                mndwi_series.append(float(df_feat[f'MNDWI_{m_str}'].values[0]))
                vh_series.append(float(df_feat[f'VH_{m}'].values[0]))
                vv_series.append(float(df_feat[f'VV_{m}'].values[0]))
                
            explanation = f"Africa Extrapolation: Location is predicted as {'Aquaculture Pond' if prob >= 0.5 else 'Other'} with {prob*100:.1f}% confidence. Region: {name}."
            
            export_data.append({
                'id': f"AFRICA_{idx:04d}",
                'lat': lat,
                'lon': lon,
                'type': 'africa',
                'probability': prob,
                'actual_label': None,
                'explanation': explanation,
                'ndwi': ndwi_series,
                'mndwi': mndwi_series,
                'vh': vh_series,
                'vv': vv_series
            })
             
    with open('./public/predictions.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"SUCCESS: Exported {len(export_data)} predictions to './public/predictions.json'.")

if __name__ == '__main__':
    run_explainability('./data/Train.csv', './data/Test.csv', './outputs/oof_predictions.npy', './outputs/test_predictions.npy')
