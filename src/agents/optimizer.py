import os
import json
import pandas as pd
import numpy as np
from src.features import extract_all_features
from src.train import run_pipeline

class ZindiOptimizerAgent:
    """
    Runs experiments by applying Zindi recommendations and measuring spatial-CV AUC score changes.
    """
    def __init__(self, actionable_tips_path='./outputs/extracted_actionable_tips.json'):
        self.actionable_tips_path = actionable_tips_path
        
    def run_experiments(self):
        print("Optimizer Agent: Starting optimization run...")
        
        # Load datasets
        train_path = './data/Train.csv'
        test_path = './data/Test.csv'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("Optimizer Agent: Error - Raw datasets not found in ./data/. Optimization run aborted.")
            return
            
        train_orig = pd.read_csv(train_path)
        test_orig = pd.read_csv(test_path)
        
        # Parse baseline features
        train_feat = extract_all_features(train_orig)
        test_feat = extract_all_features(test_orig)
        
        target_col = 'target'
        for col in train_feat.columns:
            if col.lower() in ['target', 'label', 'pond', 'class']:
                target_col = col
                break
                
        # 1. Run baseline model to get baseline CV score
        print("\n--- Running Baseline Model ---")
        baseline_artifacts = run_pipeline(train_feat, test_feat, target_col=target_col)
        baseline_auc = baseline_artifacts['oof_auc']
        print(f"Baseline Spatial-CV AUC: {baseline_auc:.5f}")
        
        # Load tips
        if not os.path.exists(self.actionable_tips_path):
            print("Optimizer Agent: No actionable recommendations found. Exiting.")
            return
            
        with open(self.actionable_tips_path, 'r') as f:
            tips = json.load(f)
            
        best_auc = baseline_auc
        best_config = {
            'features_to_exclude': ['id', 'target', 'groups'],
            'xgb_max_depth': 5,
            'shuffle_dataset': False
        }
        
        results = []
        
        # 2. Iterate and evaluate each recommendation
        for tip in tips:
            print(f"\n--- Evaluating Experiment: {tip['id']} ---")
            print(f"Rationale: {tip['rationale']}")
            
            # Setup config for this run
            run_config = best_config.copy()
            
            # Apply configuration parameters
            if tip['id'] == 'exclude_coords':
                run_config['features_to_exclude'] = tip['value']
            elif tip['id'] == 'xgb_depth_limit':
                run_config['xgb_max_depth'] = tip['value']
            elif tip['id'] == 'leakage_shuffling':
                run_config['shuffle_dataset'] = tip['value']
                
            # Copy dataframes
            exp_train = train_feat.copy()
            exp_test = test_feat.copy()
            
            if run_config['shuffle_dataset']:
                # Shuffle dataset to test if sorting leakage was present
                exp_train = exp_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
                
            try:
                # We mock running the pipeline with adjusted parameters
                # In train.py we would load the depth or features from parameters.
                artifacts = run_pipeline(
                    exp_train, 
                    exp_test, 
                    target_col=target_col, 
                    features_to_exclude=run_config['features_to_exclude']
                )
                run_auc = artifacts['oof_auc']
                print(f"Experiment {tip['id']} Spatial-CV AUC: {run_auc:.5f}")
                
                # Check for improvement
                diff = run_auc - baseline_auc
                improved = run_auc > best_auc
                if improved:
                    best_auc = run_auc
                    best_config = run_config.copy()
                    print(f"--> IMPROVEMENT FOUND! New Best AUC: {best_auc:.5f}")
                    
                results.append({
                    'id': tip['id'],
                    'oof_auc': run_auc,
                    'auc_diff': diff,
                    'improved': improved
                })
            except Exception as e:
                print(f"Error executing experiment {tip['id']}: {e}")
                results.append({
                    'id': tip['id'],
                    'error': str(e),
                    'improved': False
                })
                
        # 3. Save final experiment optimization report
        report = {
            'baseline_auc': baseline_auc,
            'best_auc': best_auc,
            'best_config': best_config,
            'experiments': results
        }
        
        output_path = './outputs/optimization_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nOptimizer Agent: Run complete. Best Spatial-CV AUC: {best_auc:.5f}. Report saved to '{output_path}'.")

if __name__ == '__main__':
    optimizer = ZindiOptimizerAgent()
    optimizer.run_experiments()
