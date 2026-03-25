import sys
import os
import time

# Import main functions from our scripts
from train_baseline import train_baseline
from train_abstention import train_abstention
from train_experiments import run_all_experiments
from evaluation import run_comprehensive_evaluation
from plots import plot_training_curves, plot_risk_coverage, plot_hardware_metrics


def print_header(phase_num, title):
    print("\n" + "#" * 90)
    print(f"# PHASE {phase_num}: {title.upper()}")
    print("#" * 90)
    time.sleep(1)


def main():
    """
    Master orchestrator script to run the exact 5-level pipeline:
    1. Train Baseline
    2. Train Abstention
    3. Run Hardware Simulations
    4. Evaluate All Models
    5. Generate Plots
    """
    
    os.makedirs("results", exist_ok=True)
    
    total_start = time.time()
    
    try:
        # ---------------------------------------------------------
        # PHASE 1: Baseline Training
        # ---------------------------------------------------------
        print_header(1, "Baseline Training")
        best_val_loss = train_baseline()
        if best_val_loss > 0.5:
            print("WARNING: Baseline training did not converge well. Val loss > 0.5")
            
        # ---------------------------------------------------------
        # PHASE 2: Abstention Training
        # ---------------------------------------------------------
        print_header(2, "Abstention Training (DAC Loss)")
        best_val_loss = train_abstention()
        if best_val_loss > 0.5:
            print("WARNING: Abstention training did not converge well. Val loss > 0.5")
            
        # ---------------------------------------------------------
        # PHASE 3: Hardware Simulation Experiments
        # ---------------------------------------------------------
        print_header(3, "Hardware Simulation Experiments")
        run_all_experiments()
        
        # ---------------------------------------------------------
        # PHASE 4: Comprehensive Evaluation
        # ---------------------------------------------------------
        print_header(4, "Comprehensive Evaluation on Test Set")
        if not os.path.exists("baseline_model.pth") or not os.path.exists("abstention_model.pth"):
            print("ERROR: Models missing. Evaluating what is available.")
        run_comprehensive_evaluation()
        
        # ---------------------------------------------------------
        # PHASE 5: Visualization & Plotting
        # ---------------------------------------------------------
        print_header(5, "Visualization & Plotting")
        plot_training_curves()
        plot_risk_coverage()
        plot_hardware_metrics()
        
        # ---------------------------------------------------------
        # DONE
        # ---------------------------------------------------------
        total_time = time.time() - total_start
        print("\n" + "=" * 90)
        print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60:.1f} MINUTES")
        print("Results and plots saved to the 'results/' directory.")
        print("=" * 90)
        
    except Exception as e:
        print("\n" + "!" * 90)
        print(f"PIPELINE FAILED with error: {str(e)}")
        print("!" * 90)
        sys.exit(1)


if __name__ == "__main__":
    main()
