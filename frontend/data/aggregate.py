import json
import os
import pandas as pd

def parse_csv_to_dict(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    df = pd.read_csv(filepath)
    return df.to_dict(orient='records')

def aggregate_results():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    
    # 1. Final Results
    final_results = parse_csv_to_dict(os.path.join(data_dir, 'final_results.csv'))
    
    # 2. Baseline Metrics
    baseline_metrics = parse_csv_to_dict(os.path.join(data_dir, 'baseline_training_metrics.csv'))
    
    # 3. Abstention Metrics
    abstention_metrics = parse_csv_to_dict(os.path.join(data_dir, 'abstention_training_metrics.csv'))
    
    # 4. Experiment Metrics
    experiment_metrics = {}
    for i in range(1, 5):
        experiment_metrics[f'exp_{i}'] = parse_csv_to_dict(os.path.join(data_dir, f'experiment_{i}_metrics.csv'))

    results_json = {
        'final_results': final_results,
        'baseline_metrics': baseline_metrics,
        'abstention_metrics': abstention_metrics,
        'experiment_metrics': experiment_metrics
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=4)
        
    print(f"Results aggregated to {output_path}")

if __name__ == "__main__":
    aggregate_results()
