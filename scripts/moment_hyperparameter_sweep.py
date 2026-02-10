"""
Hyperparameter sweep experiment for moment matching method.

This script runs moment matching experiments across different numbers of moments
to determine the optimal hyperparameter value. For each moment count, multiple
trials are run and statistics are collected.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_handling import (
    load_hyperfines_and_susceptibility_tensor,
    load_observed_pseudocontact_shift_data
)
from optimizers import MomentMatchingFitter, generate_random_chi_diagonal

sns.set_theme(style="whitegrid")


def run_moment_sweep(
    moment_range: list,
    n_trials_per_moment: int = 10,
    max_iters: int = 100,
    tol: float = 1e-8,
    chi_iso: float = 0.18013,
    use_standardized: bool = True,
    base_seed: int = 42,
    verbose: bool = False
):
    """
    Run moment matching experiments across different moment counts.
    
    Parameters
    ----------
    moment_range : list
        List of moment counts to test (e.g., [2, 4, 6, 8, 10, 12])
    n_trials_per_moment : int
        Number of trials to run for each moment count
    max_iters : int
        Maximum iterations per trial
    tol : float
        Convergence tolerance
    chi_iso : float
        Isotropic susceptibility
    use_standardized : bool
        Whether to use standardized moments
    base_seed : int
        Base random seed
    verbose : bool
        Print progress information
        
    Returns
    -------
    dict
        Results dictionary containing all trial data
    """
    # Load data once
    print("Loading experimental data...")
    hyperfines_dict, chi_true = load_hyperfines_and_susceptibility_tensor()
    delta_pc_dict = load_observed_pseudocontact_shift_data()
    
    # Store results for each moment count
    results_by_moment = {}
    
    for max_moment in moment_range:
        print(f"\n{'='*60}")
        print(f"Testing max_moment = {max_moment}")
        print(f"{'='*60}")
        
        chi_errors = []
        execution_times = []
        trial_results = []
        
        for trial in range(n_trials_per_moment):
            seed = base_seed + trial
            np.random.seed(seed)
            
            if verbose:
                print(f"  Trial {trial + 1}/{n_trials_per_moment} (seed={seed})...", end=" ")
            
            # Generate random chi initialization
            chi_init = generate_random_chi_diagonal(chi_iso=chi_iso, seed=seed)
            
            # Create fitter
            fitter = MomentMatchingFitter(
                hyperfines_dict=hyperfines_dict,
                delta_pc_dict=delta_pc_dict,
                chi_iso=chi_iso,
                chi_init=chi_init,
                max_iters=max_iters,
                tol=tol,
                optimizer='trust-constr',
                max_moment=max_moment,
                use_standardized=use_standardized
            )
            
            # Time the fitting
            start_time = time.time()
            result = fitter.fit(verbose=False)
            elapsed_time = time.time() - start_time
            
            # Compute chi error (Frobenius norm)
            chi_error = np.linalg.norm(result['chi'] - chi_true, ord='fro')
            
            chi_errors.append(chi_error)
            execution_times.append(elapsed_time)
            trial_results.append({
                'seed': seed,
                'chi': result['chi'],
                'chi_error': chi_error,
                'elapsed_time': elapsed_time,
                'converged': result['converged'],
                'n_iterations': result['n_iterations']
            })
            
            if verbose:
                print(f"χ_error={chi_error:.3e}, time={elapsed_time:.2f}s")
        
        # Compute statistics for this moment count
        results_by_moment[max_moment] = {
            'trials': trial_results,
            'chi_errors': np.array(chi_errors),
            'execution_times': np.array(execution_times),
            'best_chi_error': np.min(chi_errors),
            'mean_chi_error': np.mean(chi_errors),
            'std_chi_error': np.std(chi_errors),
            'mean_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
        }
        
        print(f"\n  Results for max_moment={max_moment}:")
        print(f"    Best χ error:  {results_by_moment[max_moment]['best_chi_error']:.6e}")
        print(f"    Mean χ error:  {results_by_moment[max_moment]['mean_chi_error']:.6e} "
              f"± {results_by_moment[max_moment]['std_chi_error']:.6e}")
        print(f"    Mean time:     {results_by_moment[max_moment]['mean_execution_time']:.2f}s "
              f"± {results_by_moment[max_moment]['std_execution_time']:.2f}s")
    
    # Package all results
    results = {
        'moment_range': moment_range,
        'n_trials_per_moment': n_trials_per_moment,
        'results_by_moment': results_by_moment,
        'chi_true': chi_true,
        'use_standardized': use_standardized,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_iters': max_iters,
            'tol': tol,
            'chi_iso': chi_iso,
            'base_seed': base_seed
        }
    }
    
    return results


def plot_moment_sweep_results(results: dict, output_dir: Path):
    """
    Plot hyperparameter sweep results.
    
    Generates a figure with three subplots:
    1. Best chi error vs number of moments
    2. Mean chi error (with std dev bars) vs number of moments
    3. Mean execution time (with std dev bars) vs number of moments
    
    Parameters
    ----------
    results : dict
        Results from run_moment_sweep()
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    moment_range = results['moment_range']
    results_by_moment = results['results_by_moment']
    use_standardized = results['use_standardized']
    n_trials = results['n_trials_per_moment']
    
    best_chi_errors = [results_by_moment[m]['best_chi_error'] for m in moment_range]
    mean_chi_errors = [results_by_moment[m]['mean_chi_error'] for m in moment_range]
    std_chi_errors = [results_by_moment[m]['std_chi_error'] for m in moment_range]
    mean_times = [results_by_moment[m]['mean_execution_time'] for m in moment_range]
    std_times = [results_by_moment[m]['std_execution_time'] for m in moment_range]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    # =========================================================================
    # Plot 1: Best and mean chi error (with error bars)
    # =========================================================================
    # Note: Using linear scale for mean±std to show symmetric error bars correctly
    axes[0].plot(moment_range, best_chi_errors, 'o-', linewidth=2, markersize=8,
                 color='darkblue', label='Best χ error', zorder=3)
    axes[0].errorbar(moment_range, mean_chi_errors, yerr=std_chi_errors,
                     fmt='s-', linewidth=2, markersize=8, capsize=5, capthick=2,
                     color='darkgreen', ecolor='lightgreen', 
                     label=f'Mean ± Std (n={n_trials})', zorder=2)
        
    axes[0].set_xlabel('Number of Moments', fontsize=12)
    axes[0].set_ylabel('χ Frobenius Error', fontsize=12)
    axes[0].set_title('χ Error vs Moment Count', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(moment_range)
    
    # Highlight minimum best
    min_idx = np.argmin(best_chi_errors)
    axes[0].plot(moment_range[min_idx], best_chi_errors[min_idx], 
                 '*', markersize=20, color='red', zorder=5,
                 label=f'Optimal (best): {moment_range[min_idx]} moments')
    
    # Highlight minimum mean (only if different from best)
    min_mean_idx = np.argmin(mean_chi_errors)
    if min_mean_idx != min_idx:
        axes[0].plot(moment_range[min_mean_idx], mean_chi_errors[min_mean_idx],
                     '*', markersize=20, color='orange', zorder=5,
                     label=f'Optimal (mean): {moment_range[min_mean_idx]} moments')
    
    axes[0].legend(fontsize=10)    
    
    # =========================================================================
    # Plot 2: Mean Execution Time (with error bars)
    # =========================================================================
    axes[1].errorbar(moment_range, mean_times, yerr=std_times,
                     fmt='s-', linewidth=2, markersize=8, capsize=5, capthick=2,
                     color='darkorange', ecolor='lightsalmon',
                     label=f'Mean ± Std (n={n_trials})')
    axes[1].set_xlabel('Number of Moments', fontsize=12)
    axes[1].set_ylabel('Execution Time (seconds)', fontsize=12)
    axes[1].set_title('Computation Time vs Moment Count', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(moment_range)
    axes[1].legend(fontsize=10)
    
    # Overall title
    moment_type = 'Standardized' if use_standardized else 'Centralized'
    fig.suptitle(f'Moment Matching Hyperparameter Sweep ({moment_type} Moments)', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'moment_hyperparameter_sweep.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_dir / 'moment_hyperparameter_sweep.png'}")
    
    # =========================================================================
    # Export summary table
    # =========================================================================
    summary_data = []
    for max_moment in moment_range:
        data = results_by_moment[max_moment]
        summary_data.append({
            'Max Moments': max_moment,
            'Best χ Error': f"{data['best_chi_error']:.6e}",
            'Mean χ Error': f"{data['mean_chi_error']:.6e}",
            'Std χ Error': f"{data['std_chi_error']:.6e}",
            'Mean Time (s)': f"{data['mean_execution_time']:.2f}",
            'Std Time (s)': f"{data['std_execution_time']:.2f}",
        })
    
    import pandas as pd
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'moment_sweep_summary.csv', index=False)
    
    print(f"\n{'='*80}")
    print("MOMENT HYPERPARAMETER SWEEP SUMMARY")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"Optimal moment count (by best χ error): {moment_range[min_idx]}")
    print(f"Optimal moment count (by mean χ error): {moment_range[min_mean_idx]}")
    print(f"{'='*80}\n")


def main():
    """Run moment hyperparameter sweep experiment."""
    print("="*80)
    print("MOMENT MATCHING HYPERPARAMETER SWEEP")
    print("="*80)
    
    # Configuration
    moment_range = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 45]  # Moment counts to test
    n_trials_per_moment = 20  # Trials per configuration
    
    print(f"\nConfiguration:")
    print(f"  Moment range: {moment_range}")
    print(f"  Trials per moment: {n_trials_per_moment}")
    print(f"  Total experiments: {len(moment_range) * n_trials_per_moment}")
    
    # Run sweep
    results = run_moment_sweep(
        moment_range=moment_range,
        n_trials_per_moment=n_trials_per_moment,
        max_iters=100,
        tol=1e-8,
        chi_iso=0.18013,
        use_standardized=True,
        base_seed=42,
        verbose=True
    )
    
    # Save results
    output_dir = Path("results/moment_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"moment_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_file}")
    
    with open("results/moment_sweep/moment_sweep_20260204_142254.pkl", 'rb') as f:
        results = pickle.load(f)
    
    # Generate plots
    output_dir = Path("results/moment_sweep")
    plot_moment_sweep_results(results, output_dir)
    
    print("\nHyperparameter sweep complete!")


if __name__ == "__main__":
    main()
