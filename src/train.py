import os
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_spatial_groups(df, n_clusters=5, seed=42):
    """
    Groups coordinates (latitude, longitude) using KMeans to create spatial clusters
    for spatial cross-validation.
    """
    coords = df[['latitude', 'longitude']].copy()
    # Normalize coords for KMeans
    coords_mean = coords.mean()
    coords_std = coords.std() + 1e-6
    coords_scaled = (coords - coords_mean) / coords_std
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    groups = kmeans.fit_predict(coords_scaled)
    return groups

class SeedAveragedClassifier:
    """
    Wrapper to train a model across multiple seeds and average predictions.
    """
    def __init__(self, model_type, params, seeds=[42, 100, 2026, 777, 888]):
        self.model_type = model_type
        self.params = params.copy()
        self.seeds = seeds
        self.models = []
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.models = []
        for seed in self.seeds:
            # Update random seeds in params
            if self.model_type == 'lgb':
                self.params['random_state'] = seed
                self.params['feature_fraction_seed'] = seed
                self.params['bagging_seed'] = seed
                model = lgb.LGBMClassifier(**self.params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            elif self.model_type == 'xgb':
                self.params['random_state'] = seed
                model = xgb.XGBClassifier(**self.params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif self.model_type == 'cat':
                self.params['random_seed'] = seed
                model = CatBoostClassifier(**self.params)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            self.models.append(model)
            
    def predict_proba(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict_proba(X)[:, 1])
        return np.mean(preds, axis=0)

def train_cv(X, y, groups, model_type, params, seeds=[42, 100, 2026, 777, 888]):
    """
    Runs spatial GroupKFold cross-validation and records OOF predictions.
    """
    oof = np.zeros(len(X))
    test_preds_list = []
    
    gkf = GroupKFold(n_splits=5)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Calculate scale_pos_weight for imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos = neg_count / (pos_count + 1e-6)
        
        # Inject scale_pos_weight
        fold_params = params.copy()
        if model_type == 'lgb':
            fold_params['scale_pos_weight'] = scale_pos
        elif model_type == 'xgb':
            fold_params['scale_pos_weight'] = scale_pos
        elif model_type == 'cat':
            fold_params['scale_pos_weight'] = scale_pos
            
        model = SeedAveragedClassifier(model_type, fold_params, seeds)
        model.fit(X_train, y_train, X_val, y_val)
        
        val_pred = model.predict_proba(X_val)
        oof[val_idx] = val_pred
        
        # Save models of fold for final test prediction
        test_preds_list.append(model)
        
    return oof, test_preds_list

def optimize_threshold(y_true, y_pred):
    """
    Finds the decision threshold that maximizes the F1 score.
    """
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_true, (y_pred >= thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

def run_pipeline(train_df, test_df, target_col='target', features_to_exclude=['id', 'target', 'groups']):
    """
    Main orchestration function.
    """
    set_seed(42)
    
    # Check shape & target
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Target distribution:\n{train_df[target_col].value_counts(normalize=True)}")
    
    # 1. Define spatial groups
    groups = get_spatial_groups(train_df, n_clusters=5)
    train_df['groups'] = groups
    
    # Identify feature columns
    features = [c for c in train_df.columns if c not in features_to_exclude]
    print(f"Training on {len(features)} engineered features.")
    
    X = train_df[features]
    y = train_df[target_col]
    X_test = test_df[features]
    
    # Models and parameters
    lgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.02,
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': -1
    }
    
    xgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.02,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    cat_params = {
        'iterations': 800,
        'learning_rate': 0.03,
        'depth': 6,
        'eval_metric': 'AUC',
        'thread_count': -1
    }
    
    print("\n--- Training LightGBM Models ---")
    lgb_oof, lgb_models = train_cv(X, y, groups, 'lgb', lgb_params)
    lgb_auc = roc_auc_score(y, lgb_oof)
    print(f"LightGBM Spatial-CV AUC: {lgb_auc:.5f}")
    
    print("\n--- Training XGBoost Models ---")
    xgb_oof, xgb_models = train_cv(X, y, groups, 'xgb', xgb_params)
    xgb_auc = roc_auc_score(y, xgb_oof)
    print(f"XGBoost Spatial-CV AUC: {xgb_auc:.5f}")
    
    print("\n--- Training CatBoost Models ---")
    cat_oof, cat_models = train_cv(X, y, groups, 'cat', cat_params)
    cat_auc = roc_auc_score(y, cat_oof)
    print(f"CatBoost Spatial-CV AUC: {cat_auc:.5f}")
    
    # 2. Ensemble Stacking
    print("\n--- Stacking Models ---")
    oof_preds = pd.DataFrame({
        'lgb': lgb_oof,
        'xgb': xgb_oof,
        'cat': cat_oof
    })
    
    # Fit meta-learner
    meta_learner = LogisticRegression(C=1.0, random_state=42)
    meta_learner.fit(oof_preds, y)
    stacked_oof = meta_learner.predict_proba(oof_preds)[:, 1]
    
    stacked_auc = roc_auc_score(y, stacked_oof)
    print(f"Stacked Ensemble Spatial-CV AUC: {stacked_auc:.5f}")
    
    # Dynamic metric check (if probabilities are expected, AUC is key; if binary, F1 optimization)
    best_thresh, best_f1 = optimize_threshold(y, stacked_oof)
    print(f"Optimal Decision Threshold: {best_thresh:.2f}")
    print(f"Best OOF F1-Score: {best_f1:.5f}")
    
    # Predict on test set
    lgb_test_preds = np.mean([m.predict_proba(X_test) for m in lgb_models], axis=0)
    xgb_test_preds = np.mean([m.predict_proba(X_test) for m in xgb_models], axis=0)
    cat_test_preds = np.mean([m.predict_proba(X_test) for m in cat_models], axis=0)
    
    test_preds_df = pd.DataFrame({
        'lgb': lgb_test_preds,
        'xgb': xgb_test_preds,
        'cat': cat_test_preds
    })
    
    final_test_probs = meta_learner.predict_proba(test_preds_df)[:, 1]
    
    # Return all artifacts
    model_artifacts = {
        'lgb_models': lgb_models,
        'xgb_models': xgb_models,
        'cat_models': cat_models,
        'meta_learner': meta_learner,
        'features': features,
        'oof_predictions': stacked_oof,
        'test_probabilities': final_test_probs,
        'oof_auc': stacked_auc,
        'best_threshold': best_thresh
    }
    
    return model_artifacts
