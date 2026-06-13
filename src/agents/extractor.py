import os
import json

class ZindiTipExtractorAgent:
    """
    Parses crawled threads, extracts keywords, and generates structured optimization recommendations.
    """
    def __init__(self, raw_tips_path='./outputs/raw_crawled_tips.json'):
        self.raw_tips_path = raw_tips_path
        
    def extract_tips(self):
        print("Extraction Agent: Parsing raw crawled tips...")
        if not os.path.exists(self.raw_tips_path):
            print("Extraction Agent: Error - Raw tips file not found.")
            return []
            
        with open(self.raw_tips_path, 'r') as f:
            threads = json.load(f)
            
        recommendations = []
        for t in threads:
            content = t['content'].lower()
            author = t['author']
            
            # Key 1: drop coordinates / spatial overfitting
            if 'drop' in content and ('latitude' in content or 'longitude' in content or 'coordinates' in content):
                recommendations.append({
                    'id': 'exclude_coords',
                    'action': 'set_coordinates_exclusion',
                    'parameter': 'features_to_exclude',
                    'value': ['id', 'target', 'groups', 'latitude', 'longitude'],
                    'rationale': f"Suggested by {author}: Coordinates cause spatial overfitting across pilot regions."
                })
                
            # Key 2: temporal median
            if 'median' in content and ('temporal' in content or 'sentinel' in content or 'noise' in content):
                recommendations.append({
                    'id': 'use_median',
                    'action': 'enable_temporal_median',
                    'parameter': 'use_median_aggregation',
                    'value': True,
                    'rationale': f"Suggested by {author}: Median aggregation is more robust to radar speckle noise."
                })
                
            # Key 3: XGBoost max depth
            if 'xgboost' in content and 'max_depth' in content:
                recommendations.append({
                    'id': 'xgb_depth_limit',
                    'action': 'set_xgb_max_depth',
                    'parameter': 'xgb_max_depth',
                    'value': 4,
                    'rationale': f"Suggested by {author}: Limit XGBoost max_depth to 4 or 5 to prevent overfitting."
                })
                
            # Key 4: Data leakage shuffling
            if 'leakage' in content or 'shuffl' in content:
                recommendations.append({
                    'id': 'leakage_shuffling',
                    'action': 'enable_shuffling',
                    'parameter': 'shuffle_dataset',
                    'value': True,
                    'rationale': f"Suggested by {author}: Shuffling dataset before splitting prevents label leakage from sorting."
                })
                
        output_path = './outputs/extracted_actionable_tips.json'
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
            
        print(f"Extraction Agent: Extracted {len(recommendations)} actionable recommendations. Saved to '{output_path}'.")
        return recommendations

if __name__ == '__main__':
    extractor = ZindiTipExtractorAgent()
    extractor.extract_tips()
