import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configure matplotlib for high-quality academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def resolve_path(rel_path):
    # Standalone path resolver to avoid importing torch from dataset.py
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel_path)

def plot_training_curves():
    """Plot training and validation loss for all 4 experiments."""
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for exp_id in range(1, 5):
        csv_path = resolve_path(f"results/experiment_{exp_id}_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df['val_loss'], label=f'Exp {exp_id} Val Loss', 
                     marker=markers[exp_id-1], markersize=5, linewidth=2, color=colors[exp_id-1], alpha=0.85)
            
    plt.title('Validation Loss across Hardware Simulation Experiments', fontweight='bold', pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (DAC)')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(resolve_path('results/plot_training_curves.png'))
    plt.close()
    print("Saved results/plot_training_curves.png")


def plot_risk_coverage(final_results_path=None):
    """Plot Risk-Coverage comparison across the evaluated models."""
    if final_results_path is None:
        final_results_path = resolve_path("results/final_results.csv")
    if not os.path.exists(final_results_path):
        print(f"Cannot plot Risk-Coverage: {final_results_path} not found.")
        return
        
    df = pd.read_csv(final_results_path)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    markers = ['X', 'o', 's', '^', 'D', 'v']
    
    for i, row in df.iterrows():
        plt.scatter(
            row['Coverage'] * 100, 
            row['Selective Risk'] * 100, 
            label=row['Model Name'], 
            s=150, 
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            edgecolor='black',
            linewidth=1.2,
            zorder=3
        )
        
        y_offset = 12 if i % 2 == 0 else -18
        x_offset = 15 if i % 2 == 0 else -15
        
        plt.annotate(
            row['Model Name'],
            (row['Coverage'] * 100, row['Selective Risk'] * 100),
            xytext=(x_offset, y_offset), 
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
            zorder=4
        )
        
    plt.title('Risk-Coverage Tradeoff', fontweight='bold', pad=15)
    plt.xlabel('Coverage (%)')
    plt.ylabel('Selective Risk (%)')
    plt.grid(True, linestyle=':', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(resolve_path('results/plot_risk_coverage.png'))
    plt.close()
    print("Saved results/plot_risk_coverage.png")


def plot_hardware_metrics():
    """Plot average throughput and memory usage for the 4 experiments."""
    
    experiments = []
    throughputs = []
    memories = []
    
    for exp_id in range(1, 5):
        csv_path = resolve_path(f"results/experiment_{exp_id}_metrics.csv")
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
    plt.figure(figsize=(8, 5))
    bars = plt.bar(experiments, throughputs, color='#3498db', edgecolor='black', linewidth=1.2, width=0.6)
    plt.title('Average Training Throughput', fontweight='bold', pad=15)
    plt.ylabel('Samples per Second')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(throughputs)*0.02), f"{yval:.0f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
                 
    plt.tight_layout()
    plt.savefig(resolve_path('results/plot_hardware_throughput.png'))
    plt.close()
    print("Saved results/plot_hardware_throughput.png")
    
    # Plot Memory
    plt.figure(figsize=(8, 5))
    bars = plt.bar(experiments, memories, color='#e74c3c', edgecolor='black', linewidth=1.2, width=0.6)
    plt.title('Peak Process Memory', fontweight='bold', pad=15)
    plt.ylabel('Memory (MB)')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(max(memories), 1)*0.02), f"{yval:.1f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
                 
    plt.tight_layout()
    plt.savefig(resolve_path('results/plot_hardware_memory.png'))
    plt.close()
    print("Saved results/plot_hardware_memory.png")

    
if __name__ == "__main__":
    os.makedirs(resolve_path("results"), exist_ok=True)
    plot_training_curves()
    plot_risk_coverage()
    plot_hardware_metrics()