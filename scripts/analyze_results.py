"""
Analysis and visualization tools for experiment results.

This module provides functions to:
- Load and analyze saved experiment results
- Generate comparison plots
- Create animations of optimization process
- Export summary statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")


def load_results(results_file: str) -> Dict:
    """
    Load experiment results from pickle file.
    
    Parameters
    ----------
    results_file : str
        Path to results pickle file
        
    Returns
    -------
    dict
        Loaded experiment results
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_convergence_curves(results: Dict, save_path: Optional[str] = None):
    """
    Plot convergence curves for all trials.
    
    Parameters
    ----------
    results : dict
        Experiment results
    save_path : str, optional
        Path to save figure
    """
    trial_results = results['trials']['results']
    best_idx = results['best_trial_idx']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, result in enumerate(trial_results):
        alpha = 0.3 if i != best_idx else 1.0
        linewidth = 1 if i != best_idx else 3
        color = 'blue' if i != best_idx else 'red'
        label = f'Trial {best_idx+1} (best)' if i == best_idx else None
        
        ax.plot(result['loss_history'], alpha=alpha, linewidth=linewidth, 
                color=color, label=label)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Convergence Across All Trials', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if best_idx is not None:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    
    plt.show()
    

def create_animation(
    frame_data: List[Dict],
    output_file: str='optimization_animation.mp4',
    fps: int = 2
):
    """
    Create animation showing optimization progress.
    
    Parameters
    ----------
    frame_data : list[dict]
        Frame-by-frame optimization data
    output_file : str
        Output filename for animation
    fps : int
        Frames per second
    """
    print(f"Creating animation with {len(frame_data)} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter([], [], s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Get axis limits
    all_observed = frame_data[0]['observed']
    all_predicted = np.concatenate([f['predicted'] for f in frame_data])
    axis_min = min(all_observed.min(), all_predicted.min()) * 1.1
    axis_max = max(all_observed.max(), all_predicted.max()) * 1.1
    
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel('Observed δ_pc (ppm)', fontsize=14)
    ax.set_ylabel('Predicted δ_pc (ppm)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Perfect fit line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', 
            linewidth=2, label='Perfect fit', zorder=1)
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,
    
    def update(frame_idx):
        frame = frame_data[frame_idx]
        
        # Update scatter plot
        points = np.column_stack([frame['observed'], frame['predicted']])
        scatter.set_offsets(points)
        
        # Update title
        if frame['step'] == 'initial':
            title = f"Initial: Random Assignment\nMSE = {frame['loss']:.4e}"
        elif frame['step'] == 'chi_optimized':
            title = f"Iteration {frame['iteration']}: After χ Optimization\nMSE = {frame['loss']:.4e}"
        else:
            title = f"Iteration {frame['iteration']}: After Assignment (Hungarian)\nMSE = {frame['loss']:.4e}"
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=12, loc='upper left')
        
        return scatter,
    
    ani = FuncAnimation(
        fig, update, frames=len(frame_data),
        init_func=init, blit=False, repeat=True,
        interval=500
    )
    
    ani.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
    print(f"Animation saved to {output_file}")
    plt.close()


def create_moment_matching_animation(
    frame_data: List[Dict],
    output_file: str = 'moment_matching_animation.mp4',
    fps: int = 2,
    use_standardized: bool = False
):
    """
    Create dual-panel animation for moment matching optimization.
    
    Left panel: Shifts (observed vs predicted)
    Right panel: Moments (observed vs predicted)
    
    Parameters
    ----------
    frame_data : list[dict]
        Frame-by-frame optimization data from MomentMatchingFitter
    output_file : str
        Output filename for animation
    fps : int
        Frames per second
    use_standardized : bool
        Whether standardized moments were used
    """
    print(f"Creating moment matching animation with {len(frame_data)} frames...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT PANEL: Shifts
    scatter_shifts = ax1.scatter([], [], s=100, alpha=0.7, c='blue', 
                                edgecolors='black', linewidth=1.5)
    
    all_observed_shifts = frame_data[0]['observed_shifts']
    all_predicted_shifts = np.concatenate([f['predicted_shifts'] for f in frame_data])
    shift_min = min(all_observed_shifts.min(), all_predicted_shifts.min()) * 1.1
    shift_max = max(all_observed_shifts.max(), all_predicted_shifts.max()) * 1.1
    
    ax1.set_xlim(shift_min, shift_max)
    ax1.set_ylim(shift_min, shift_max)
    ax1.set_xlabel('Observed δ_pc (ppm)', fontsize=14)
    ax1.set_ylabel('Predicted δ_pc (ppm)', fontsize=14)
    ax1.set_title('Pseudocontact Shifts', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.plot([shift_min, shift_max], [shift_min, shift_max], 'r--', 
            linewidth=2, label='Perfect fit', zorder=1)
    ax1.legend(fontsize=12, loc='upper left')
    
    # RIGHT PANEL: Moments
    n_moments = len(frame_data[0]['observed_moments'])
    moment_indices = np.arange(1, n_moments + 1)
    
    line_observed, = ax2.plot([], [], 'o-', markersize=10, linewidth=2.5, 
                             color='red', label='Observed', zorder=3)
    line_predicted, = ax2.plot([], [], 's--', markersize=8, linewidth=2, 
                              color='blue', label='Predicted', alpha=0.7, zorder=2)
    
    all_moments = np.concatenate([
        f['observed_moments'] for f in frame_data] + 
        [f['predicted_moments'] for f in frame_data]
    )
    moment_min = all_moments.min() * 1.1 if all_moments.min() < 0 else all_moments.min() * 0.9
    moment_max = all_moments.max() * 1.1
    
    ax2.set_xlim(0.5, n_moments + 0.5)
    ax2.set_ylim(moment_min, moment_max)
    ax2.set_xlabel('Moment Order', fontsize=14)
    
    if use_standardized:
        ax2.set_ylabel('Standardized Moment Value', fontsize=14)
        moment_labels = ['μ', 'σ']
        moment_labels += [f'M{i}' for i in range(3, n_moments + 1)]
        ax2.set_title('Standardized Moments', fontsize=15, fontweight='bold')
    else:
        ax2.set_ylabel('Moment Value', fontsize=14)
        moment_labels = [f'M{i}' for i in range(1, n_moments + 1)]
        ax2.set_title('Centralized Moments', fontsize=15, fontweight='bold')
    
    ax2.set_xticks(moment_indices)
    ax2.set_xticklabels(moment_labels)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=12, loc='upper left')
    
    # Overall title
    fig.suptitle('', fontsize=16, fontweight='bold', y=0.98)
    
    def init():
        scatter_shifts.set_offsets(np.empty((0, 2)))
        line_observed.set_data([], [])
        line_predicted.set_data([], [])
        return scatter_shifts, line_observed, line_predicted
    
    def update(frame_idx):
        frame = frame_data[frame_idx]
        
        # Update shifts scatter
        points = np.column_stack([frame['observed_shifts'], frame['predicted_shifts']])
        scatter_shifts.set_offsets(points)
        
        # Update moments lines
        line_observed.set_data(moment_indices, frame['observed_moments'])
        line_predicted.set_data(moment_indices, frame['predicted_moments'])
        
        # Update overall title
        if frame['step'] == 'initial':
            title = f"Initial State: χ = diag([0, 0, 0])\nLoss = {frame['loss']:.4e}"
        else:
            title = f"Optimization Step {frame_idx}\nLoss = {frame['loss']:.4e}"
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        return scatter_shifts, line_observed, line_predicted
    
    ani = FuncAnimation(
        fig, update, frames=len(frame_data),
        init_func=init, blit=False, repeat=True,
        interval=500
    )
    
    plt.tight_layout()
    ani.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
    print(f"Animation saved to {output_file}")
    plt.close()


def export_summary_table(results: Dict, output_file: str):
    """
    Export summary statistics to CSV.
    
    Parameters
    ----------
    results : dict
        Experiment results
    output_file : str
        Output CSV filename
    """
    metrics = results['trials']['metrics']
    
    df = pd.DataFrame(metrics)
    df['trial'] = range(1, len(metrics) + 1)
    
    # Reorder columns
    cols = ['trial', 'final_loss', 'chi_frobenius_error', 'n_iterations', 
            'converged', 'constraints_valid', 'seed']
    df = df[cols]
    
    df.to_csv(output_file, index=False, float_format='%.6e')
    print(f"Summary table saved to {output_file}")


def plot_noise_vs_chi_error(results: Dict, save_path: Optional[str] = None):
    """
    Plot best chi error and shift error vs noise level with chi_init values.
    
    Parameters
    ----------
    results : dict
        Noise sensitivity experiment results
    save_path : str, optional
        Path to save figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    noise_levels = [r['noise_level'] * 100 for r in results['level_results']]
    
    # Extract errors and log-transform manually for true log scale display
    best_chi_errors = [np.log10(r['best_chi_error']) for r in results['level_results']]
    best_shift_errors = [np.log10(r['best_loss']) for r in results['level_results']]  # MSE in shifts
    
    # Get chi_init diagonals from best trial at each noise level
    chi_init_labels = []
    for level_result in results['level_results']:
        best_idx = level_result['best_trial_idx']
        best_trial = level_result['trials'][best_idx]
        chi_diag = best_trial['chi_init'].diagonal()
        chi_init_labels.append(f"({chi_diag[0]:.4f}, {chi_diag[1]:.4f}, {chi_diag[2]:.4f})")

    
    # Chi errors on left y-axis
    line1 = ax1.plot(noise_levels, best_chi_errors, 'o-', linewidth=2, markersize=8,
                     label='Best χ error (Frobenius)', color='blue')
    
    # Shift errors on right y-axis
    line3 = ax1.plot(noise_levels, best_shift_errors, 'd-', linewidth=2, markersize=8,
                          label='Best shift error (MSE)', color='red')
    
    # Create x-tick labels with noise level and chi_init values
    x_labels = [f"{nl:.1f}%\n{chi_init}" for nl, chi_init in zip(noise_levels, chi_init_labels)]
    ax1.set_xticks(noise_levels)
    ax1.set_xticklabels(x_labels, fontsize=10)
    
    ax1.set_xlabel('Noise Level (%) and best χ_init (χ_x, χ_y, χ_z)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$log_{10}($error$)$', fontsize=12)
    ax1.set_title('χ and Shift Recovery Error vs Initialization Noise', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=12, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def export_noise_summary(results: Dict, output_file: str):
    """
    Export noise sensitivity results to CSV file.
    
    Parameters
    ----------
    results : dict
        Noise sensitivity experiment results
    output_file : str
        Path to output CSV file
    """
    rows = []
    for level_result in results['level_results']:
        rows.append({
            'noise_level_pct': level_result['noise_level'] * 100,
            'best_loss': level_result['best_loss'],
            'best_chi_error': level_result['best_chi_error'],
            'mean_loss': level_result['mean_loss'],
            'mean_chi_error': level_result['mean_chi_error'],
            'std_chi_error': level_result['std_chi_error'],
            'n_trials': len(level_result['trials'])
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nNoise sensitivity summary exported to: {output_file}")
    print(df.to_string(index=False))


def main():
    """Example usage of analysis tools."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.pkl>")
        return
    
    results_file = sys.argv[1]
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    
    output_dir = Path("results/figures")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    export_summary_table(results, output_file=output_dir / "summary_table.csv")
    
    # Create animation from best trial
    best_idx = results['best_trial_idx']
    best_result = results['trials']['results'][best_idx]
    experiment_name = best_result["experiment_name"]
    print(f"\nBest trial index: {best_idx}, Experiment: {experiment_name}")
    
    if 'frame_data' in best_result and len(best_result['frame_data']) > 0:
        print("\nCreating animation from best trial...")
        animation_file = output_dir / f"best_trial_{best_idx+1}_{experiment_name}_animation.mp4"
        try:
            if experiment_name == "alternating_optimization":
                create_animation(
                    best_result['frame_data'],
                    output_file=str(animation_file),
                    fps=2
                )
            elif experiment_name == "moment_matching":
                create_moment_matching_animation(
                    best_result['frame_data'],
                    output_file=str(animation_file),
                    fps=2,
                    use_standardized=True
                )
        except Exception as e:
            print(f"Warning: Animation creation failed: {e}")
    else:
        print("\nNote: No frame data available for animation (run experiment with frame tracking enabled)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()