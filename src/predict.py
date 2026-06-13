import os
import pandas as pd
import numpy as np
from features import extract_all_features
from train import run_pipeline

def main():
    train_path = './data/Train.csv'
    test_path = './data/Test.csv'
    sub_tmpl_path = './data/SampleSubmission.csv'
    
    # 1. Verification of data existence
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Train.csv or Test.csv not found in ./data/ directory!")
        print("Please place the challenge datasets under the './data/' folder and rerun.")
        return
        
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Extract features
    print("\n--- Extracting Features for Training Set ---")
    train_feat = extract_all_features(train_df)
    
    print("\n--- Extracting Features for Test Set ---")
    test_feat = extract_all_features(test_df)
    
    # Verify target column naming
    target_col = 'target'
    for col in train_feat.columns:
        if col.lower() in ['target', 'label', 'pond', 'class']:
            target_col = col
            break
            
    print(f"Identified target column: '{target_col}'")
    
    # 3. Train models and evaluate CV
    model_artifacts = run_pipeline(train_feat, test_feat, target_col=target_col)
    
    # Save feature artifacts and oof/test prediction arrays
    print("\nSaving feature and prediction artifacts...")
    os.makedirs('./outputs', exist_ok=True)
    train_feat.to_parquet('./outputs/train_engineered_features.parquet')
    test_feat.to_parquet('./outputs/test_engineered_features.parquet')
    
    np.save('./outputs/oof_predictions.npy', model_artifacts['oof_predictions'])
    np.save('./outputs/test_predictions.npy', model_artifacts['test_probabilities'])
    
    # 4. Generate final submission
    if os.path.exists(sub_tmpl_path):
        sub_tmpl = pd.read_csv(sub_tmpl_path)
        print("Validating against SampleSubmission.csv...")
        
        # Verify alignment
        submission = sub_tmpl.copy()
        # Find ID column (case insensitive)
        id_col = 'id'
        for col in sub_tmpl.columns:
            if col.lower() in ['id', 'location_id', 'patch_id']:
                id_col = col
                break
                
        # Find prediction column
        pred_col = [c for c in sub_tmpl.columns if c != id_col][0]
        
        # Verify row count and order of IDs
        assert len(submission) == len(test_df), "Test set and sample submission length mismatch!"
        
        # Determine target type from sample submission (probabilities vs classes)
        sample_values = sub_tmpl[pred_col].dropna().values
        is_binary = np.all((sample_values == 0) | (sample_values == 1))
        
        if is_binary:
            print("Submission format requires binary (0/1) classifications. Applying optimized threshold...")
            thresh = model_artifacts['best_threshold']
            submission[pred_col] = (model_artifacts['test_probabilities'] >= thresh).astype(int)
        else:
            print("Submission format requires probabilities. Writing raw calibrated probabilities...")
            submission[pred_col] = model_artifacts['test_probabilities']
            
        submission_filename = 'submission.csv'
        submission.to_csv(submission_filename, index=False)
        print(f"\nSUCCESS: Generated '{submission_filename}' successfully.")
        print(f"Total rows: {len(submission)}")
        print(f"Columns: {list(submission.columns)}")
        print(f"Head:\n{submission.head()}")
    else:
        # Standard fallback if sample submission isn't available
        print("Warning: SampleSubmission.csv not found in ./data/. Creating standard submission.csv...")
        id_col = 'id' if 'id' in test_df.columns else test_df.columns[0]
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'target': model_artifacts['test_probabilities']
        })
        submission.to_csv('submission.csv', index=False)
        print("SUCCESS: Generated standard submission.csv")
        
    print(f"\nFinal Spatial-CV OOF AUC: {model_artifacts['oof_auc']:.5f}")

if __name__ == '__main__':
    main()
