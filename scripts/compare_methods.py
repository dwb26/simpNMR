"""
Run both alternating optimization and moment matching experiments for comparison.

This script runs both experiments with the same number of trials and generates
comparative analysis plots.
"""

import subprocess
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

sns.set_theme(style="whitegrid")

def parity_test_in_number_of_trials():
    """Ensure both configs have the same number of trials."""
    alt_config = Path("configs/alternating_optimization_trials.yaml")
    mom_config = Path("configs/moment_matching_trials.yaml")
    import yaml
    with open(alt_config, 'r') as f:
        alt_cfg = yaml.safe_load(f)
    with open(mom_config, 'r') as f:
        mom_cfg = yaml.safe_load(f)
    n_trials_alt = alt_cfg['optimization']['n_trials']
    n_trials_mom = mom_cfg['optimization']['n_trials']
    if n_trials_alt != n_trials_mom:
        print(f"Error: Number of trials must be the same in both configs.")
        print(f"  Alternating Optimization trials: {n_trials_alt}")
        print(f"  Moment Matching trials: {n_trials_mom}")
        sys.exit(1)
        

def run_experiment(config_file: str) -> str:
    """
    Run an experiment and return path to results file.
    
    Parameters
    ----------
    config_file : str
        Name of config file (e.g., 'alternating_optimization_trials.yaml')
        
    Returns
    -------
    str
        Path to results pickle file
    """
    cmd = [sys.executable, "scripts/run_experiments.py", "--config", config_file]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running experiment:\n{result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    
    # Extract results filename from output
    for line in result.stdout.split('\n'):
        if 'Results saved to: ' in line:
            return line.split('Results saved to: ')[1].strip()
    
    raise RuntimeError("Could not find results file path in output")


def load_results(results_file: str) -> dict:
    """Load results from pickle file."""
    print(f"Loading results from: {results_file}")
    with open(results_file, 'rb') as f:
        return pickle.load(f)


def compare_methods(alternating_results: dict, moment_results: dict, output_dir: Path):
    """
    Generate comparison plots between the two methods.
    
    Focuses on chi Frobenius error as the primary accuracy metric,
    plus execution time and distribution analysis.
    
    Parameters
    ----------
    alternating_results : dict
        Results from alternating optimization
    moment_results : dict
        Results from moment matching
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    alt_metrics = alternating_results['trials']['metrics']
    mom_metrics = moment_results['trials']['metrics']
    
    # Create DataFrames
    df_alt = pd.DataFrame(alt_metrics)
    df_alt['method'] = 'Alternating Optimization'
    df_alt['trial'] = range(len(df_alt))
    
    df_mom = pd.DataFrame(mom_metrics)
    df_mom['method'] = 'Moment Matching'
    df_mom['trial'] = range(len(df_mom))
    
    # Get moment matching hyperparameters
    max_moment = moment_results['config'].get('moment_matching', {}).get('max_moment', 'N/A')
    use_standardized = moment_results['config'].get('moment_matching', {}).get('use_standardized', True)
    moment_info = f"Moments: {max_moment} ({'standardized' if use_standardized else 'centralized'})"
    
    # =========================================================================
    # SINGLE FIGURE: Chi Error KDE + Execution Time Box Plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    from scipy.stats import gaussian_kde
    
    # -------------------------------------------------------------------------
    # Left Panel: Chi Error Distribution (KDE)
    # -------------------------------------------------------------------------
    # Compute KDEs
    alt_chi_errors = df_alt['chi_frobenius_error'].values
    mom_chi_errors = df_mom['chi_frobenius_error'].values
    
    # Use log scale for KDE
    alt_log_errors = np.log10(alt_chi_errors)
    mom_log_errors = np.log10(mom_chi_errors)
    
    kde_alt = gaussian_kde(alt_log_errors)
    kde_mom = gaussian_kde(mom_log_errors)
    
    # Create evaluation points
    x_min = min(alt_log_errors.min(), mom_log_errors.min()) - 0.5
    x_max = max(alt_log_errors.max(), mom_log_errors.max()) + 0.5
    x_eval = np.linspace(x_min, x_max, 200)
    
    axes[0].plot(10**x_eval, kde_alt(x_eval), linewidth=2.5, 
                 label='Alternating Optimization', alpha=0.8)
    axes[0].plot(10**x_eval, kde_mom(x_eval), linewidth=2.5, 
                 label=f'Moment Matching ({moment_info})', alpha=0.8)
    axes[0].set_xlabel('χ Frobenius Error', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('χ Error Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Add mean and std as text annotations (positioned in upper left to avoid legend)
    alt_mean_chi = alt_chi_errors.mean()
    alt_std_chi = alt_chi_errors.std()
    mom_mean_chi = mom_chi_errors.mean()
    mom_std_chi = mom_chi_errors.std()
    
    text_str = f"Alternating: μ={alt_mean_chi:.3e}, σ={alt_std_chi:.3e}\n"
    text_str += f"Moment Match: μ={mom_mean_chi:.3e}, σ={mom_std_chi:.3e}"
    axes[0].text(0.02, 0.97, text_str, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # -------------------------------------------------------------------------
    # Right Panel: Execution Time (Box Plot)
    # -------------------------------------------------------------------------
    df_combined = pd.concat([df_alt, df_mom], ignore_index=True)
    alt_times = df_alt['elapsed_time'].values
    mom_times = df_mom['elapsed_time'].values
    
    sns.boxplot(data=df_combined, x='method', y='elapsed_time', ax=axes[1])
    axes[1].set_ylabel('Execution Time (seconds)', fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add mean markers on box plot
    for i, method_df in enumerate([df_alt, df_mom]):
        mean_val = method_df['elapsed_time'].mean()
        axes[1].plot(i, mean_val, 'D', color='red', markersize=8, zorder=5,
                     label='Mean' if i == 0 else '')
    axes[1].legend(fontsize=9)
    
    # Add mean and std as text annotations
    alt_mean_time = alt_times.mean()
    alt_std_time = alt_times.std()
    mom_mean_time = mom_times.mean()
    mom_std_time = mom_times.std()
    
    time_text = f"Alternating: μ={alt_mean_time:.2f}s, σ={alt_std_time:.2f}s\n"
    time_text += f"Moment Match: μ={mom_mean_time:.2f}s, σ={mom_std_time:.2f}s"
    axes[1].text(0.98, 0.97, time_text, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {output_dir / 'method_comparison.png'}")
    
    # =========================================================================
    # Summary statistics table
    # =========================================================================
    summary_stats = []
    
    for method_name, df_method in [('Alternating Optimization', df_alt), 
                                     ('Moment Matching', df_mom)]:
        stats = {
            'Method': method_name,
            'Best χ Error': f"{df_method['chi_frobenius_error'].min():.6e}",
            'Median χ Error': f"{df_method['chi_frobenius_error'].median():.6e}",
            'Mean χ Error': f"{df_method['chi_frobenius_error'].mean():.6e}",
            'Std χ Error': f"{df_method['chi_frobenius_error'].std():.6e}",
            'Mean Time (s)': f"{df_method['elapsed_time'].mean():.3f}",
            'Median Time (s)': f"{df_method['elapsed_time'].median():.3f}",
            'Success Rate': f"{df_method['constraints_valid'].mean()*100:.1f}%",
        }
        summary_stats.append(stats)
    
    # Add moment matching info to second row
    if len(summary_stats) > 1:
        summary_stats[1]['Method'] = f"Moment Matching ({moment_info})"
    
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(output_dir / 'comparison_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("\n")
    
    # Statistical tests on chi error
    from scipy.stats import mannwhitneyu
    
    print("="*80)
    print("STATISTICAL TESTS (Chi Frobenius Error)")
    print("="*80)
    
    stat_chi, p_chi = mannwhitneyu(alt_chi_errors, mom_chi_errors, alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  Statistic: {stat_chi:.4f}")
    print(f"  p-value: {p_chi:.6e}")
    print(f"  Significant at α=0.05: {p_chi < 0.05}")
    
    # Additional statistics
    print(f"\nMedian χ error ratio (Alt/Mom): {np.median(alt_chi_errors)/np.median(mom_chi_errors):.3f}")
    print(f"Best χ error ratio (Alt/Mom): {alt_chi_errors.min()/mom_chi_errors.min():.3f}")
    
    print(f"\nMedian time ratio (Alt/Mom): {np.median(alt_times)/np.median(mom_times):.3f}")
    
    print("\n" + "="*80)


def main():
    """Run both experiments and generate comparison."""
    print("="*80)
    print("RUNNING COMPARATIVE EXPERIMENTS")
    print("="*80)
    
    # Ensure both configs have same number of trials
    print("Checking parity in number of trials...")    
    parity_test_in_number_of_trials()
    
    # Run alternating optimization
    print("\n[1/2] Running Alternating Optimization...")
    alt_results_file = run_experiment("alternating_optimization_trials.yaml")
    
    # Run moment matching
    print("\n[2/2] Running Moment Matching...")
    mom_results_file = run_experiment("moment_matching_trials.yaml")
    
    # Load results
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    alt_results = load_results(alt_results_file)
    mom_results = load_results(mom_results_file)
    
    # Generate comparison
    output_dir = Path("results/comparison")
    compare_methods(alt_results, mom_results, output_dir)
    
    print(f"\nComparison plots saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
