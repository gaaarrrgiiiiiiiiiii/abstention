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

    
def plot_pareto_frontier_from_csv():
    """Plot the α-sweep Pareto frontier from alpha_sweep.py output."""
    csv_path = resolve_path("results/alpha_pareto.csv")
    if not os.path.exists(csv_path):
        print(f"Pareto data not found: {csv_path}. Run src/alpha_sweep.py first.")
        return

    df = pd.read_csv(csv_path)
    cmap = plt.cm.RdYlGn
    n = len(df)
    colors = [cmap(1 - i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, row in df.iterrows():
        axes[0].scatter(row["coverage"] * 100, row["selective_risk"] * 100,
                        color=colors[i], s=130, zorder=3,
                        edgecolor="black", linewidth=0.8)
        axes[0].annotate(f"α={row['alpha']:.1f}",
                         (row["coverage"] * 100, row["selective_risk"] * 100),
                         xytext=(5, 5), textcoords="offset points", fontsize=9,
                         fontweight="bold")
    axes[0].plot(df["coverage"] * 100, df["selective_risk"] * 100,
                 "--", color="gray", linewidth=1.2, alpha=0.7)
    axes[0].set_xlabel("Coverage (%)")
    axes[0].set_ylabel("Selective Risk (%)")
    axes[0].set_title("Risk-Coverage Pareto Frontier\n(by Abstention Penalty α)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(df["coverage"] * 100, df["f1"], marker="o", color="#2196F3",
                 linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.8)
    for i, row in df.iterrows():
        axes[1].annotate(f"α={row['alpha']:.1f}",
                         (row["coverage"] * 100, row["f1"]),
                         xytext=(5, -12), textcoords="offset points", fontsize=9)
    axes[1].set_xlabel("Coverage (%)")
    axes[1].set_ylabel("F1 Score (Fraud Class)")
    axes[1].set_title("F1 Score vs Coverage\n(by Abstention Penalty α)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("DAC Abstention Penalty Ablation Study", fontweight="bold",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    out = resolve_path("results/plot_pareto_frontier.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_calibration_curve(val_probs, val_labels, n_bins=10, model_name="DAC"):
    """
    Reliability diagram: compares average predicted probability vs actual
    fraction of positives within equal-width bins.

    Args:
        val_probs  : np.ndarray (N,)  — predicted fraud probability for each sample
        val_labels : np.ndarray (N,)  — ground-truth binary labels
        n_bins     : int              — number of equal-width bins
        model_name : str              — label for the legend
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_means = []
    frac_pos  = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (val_probs >= lo) & (val_probs < hi)
        if mask.sum() > 0:
            bin_means.append(float(val_probs[mask].mean()))
            frac_pos.append(float(val_labels[mask].mean()))

    bin_means = np.array(bin_means)
    frac_pos  = np.array(frac_pos)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    plt.plot(bin_means, frac_pos, marker="o", linewidth=2, color="#2196F3",
             markersize=8, markeredgecolor="black", markeredgewidth=0.8,
             label=f"{model_name} (empirical)")
    plt.fill_between(bin_means, bin_means, frac_pos, alpha=0.15, color="#F44336",
                     label="Calibration gap")
    plt.xlabel("Mean Predicted Probability (Fraud)")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Reliability Diagram — {model_name}", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    out = resolve_path(f"results/plot_calibration_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_confusion_matrix(y_true, y_pred, labels=None, model_name="Model"):
    """
    Annotated confusion matrix heatmap.
    y_pred values: 0=Legitimate, 1=Fraud, 2=Abstain
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.colors as mcolors

    if labels is None:
        labels = ["Legitimate", "Fraud", "Abstain"]

    # Build confusion matrix only for non-abstained samples if abstain present
    unique = sorted(set(y_pred) | set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=unique)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_labels = [labels[i] for i in unique] if max(unique) < len(labels) else [str(i) for i in unique]
    ax.set_xticks(range(len(unique)))
    ax.set_yticks(range(len(unique)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticklabels(tick_labels)

    thresh = cm.max() / 2.0
    for i in range(len(unique)):
        for j in range(len(unique)):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=11)

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold", pad=12)
    plt.tight_layout()
    out = resolve_path(f"results/plot_confusion_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_abstention_breakdown(y_true, y_pred):
    """
    Pie chart showing the class composition of abstained transactions
    (Legitimate vs Fraud within the abstained subset).
    """
    abstained = y_true[y_pred == 2] if hasattr(y_pred, '__len__') else np.array([])
    if len(abstained) == 0:
        print("No abstentions to plot.")
        return

    n_legit = int((abstained == 0).sum())
    n_fraud = int((abstained == 1).sum())
    total   = n_legit + n_fraud

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes  = [n_legit, n_fraud]
    labels = [
        f"Legitimate\n({n_legit:,} / {n_legit/total:.1%})",
        f"Fraud\n({n_fraud:,} / {n_fraud/total:.1%})",
    ]
    colors = ["#2196F3", "#F44336"]
    explode = (0, 0.05)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title(f"Abstained Transaction Breakdown\n({total:,} total abstentions)",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    out = resolve_path("results/plot_abstention_breakdown.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs(resolve_path("results"), exist_ok=True)
    plot_training_curves()
    plot_risk_coverage()
    plot_hardware_metrics()
    plot_pareto_frontier_from_csv()
    print("\nNote: calibration, confusion matrix, and abstention breakdown plots")
    print("      require y_true/y_pred arrays — call them from evaluation.py.")