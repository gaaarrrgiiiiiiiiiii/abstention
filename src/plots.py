import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_training_curves():
    """Plot training and validation loss for all 4 experiments."""
    plt.figure(figsize=(12, 6))
    
    for exp_id in range(1, 5):
        csv_path = f"results/experiment_{exp_id}_metrics.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['val_loss'], label=f'Exp {exp_id} Val Loss', marker='o', markersize=4)
            
    plt.title('Validation Loss across Hardware Simulation Experiments')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (DAC)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/plot_training_curves.png')
    print("Saved results/plot_training_curves.png")


def plot_risk_coverage(final_results_path="results/final_results.csv"):
    """Plot Risk-Coverage comparison across the evaluated models."""
    if not os.path.exists(final_results_path):
        print(f"Cannot plot Risk-Coverage: {final_results_path} not found.")
        return
        
    df = pd.read_csv(final_results_path)
    
    # We want a scatter plot showing Risk vs Coverage
    plt.figure(figsize=(10, 6))
    
    colors = ['gray', 'blue', 'green', 'orange', 'red', 'purple']
    markers = ['x', 'o', 's', '^', 'D', 'v']
    
    for i, row in df.iterrows():
        plt.scatter(
            row['Coverage'] * 100, 
            row['Selective Risk'] * 100, 
            label=row['Model Name'], 
            s=120, 
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)]
        )
        # Add labels next to points
        plt.annotate(
            row['Model Name'],
            (row['Coverage'] * 100, row['Selective Risk'] * 100),
            xytext=(10, 5), 
            textcoords='offset points',
            fontsize=9
        )
        
    plt.title('Risk-Coverage Tradeoff')
    plt.xlabel('Coverage (%)')
    plt.ylabel('Selective Risk (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    plt.tight_layout()
    plt.savefig('results/plot_risk_coverage.png')
    print("Saved results/plot_risk_coverage.png")


def plot_hardware_metrics():
    """Plot average throughput and memory usage for the 4 experiments."""
    
    experiments = []
    throughputs = []
    memories = []
    
    for exp_id in range(1, 5):
        csv_path = f"results/experiment_{exp_id}_metrics.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'throughput_samples_per_sec' in df.columns and ('process_memory_mb' in df.columns or 'gpu_memory_mb' in df.columns):
                mem_col = 'process_memory_mb' if 'process_memory_mb' in df.columns else 'gpu_memory_mb'
                experiments.append(f"Exp {exp_id}")
                throughputs.append(df['throughput_samples_per_sec'].mean())
                memories.append(df[mem_col].max())
                
    if not experiments:
        print("Hardware metrics not found in CSVs.")
        return
        
    # Plot Throughput
    plt.figure(figsize=(10, 5))
    plt.bar(experiments, throughputs, color='skyblue')
    plt.title('Average Training Throughput by Experiment')
    plt.ylabel('Samples per Second')
    for i, v in enumerate(throughputs):
        plt.text(i, v + (max(throughputs)*0.02), f"{v:.0f}", ha='center')
    plt.tight_layout()
    plt.savefig('results/plot_hardware_throughput.png')
    print("Saved results/plot_hardware_throughput.png")
    
    # Plot Memory
    plt.figure(figsize=(10, 5))
    plt.bar(experiments, memories, color='lightcoral')
    plt.title('Peak Process Memory by Experiment')
    plt.ylabel('Memory (MB)')
    for i, v in enumerate(memories):
        plt.text(i, v + (max(max(memories), 1)*0.02), f"{v:.1f}", ha='center')
    plt.tight_layout()
    plt.savefig('results/plot_hardware_memory.png')
    print("Saved results/plot_hardware_memory.png")

    
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_training_curves()
    plot_risk_coverage()
    plot_hardware_metrics()