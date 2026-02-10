"""
Main script for running alternating optimization experiments.

This script:
1. Loads configuration from YAML file
2. Loads experimental data
3. Runs multiple trials with different random seeds
4. Saves results to disk
"""

import argparse
import numpy as np
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_handling import (
    load_hyperfines_and_susceptibility_tensor,
    load_observed_pseudocontact_shift_data
)
from optimizers import (
    BaseAlternatingFitter,
    AlternatingOptimizationFitter,
    MomentMatchingFitter
)


def load_config(config_file: str = "configs/susc_fit.yaml") -> Dict:
    """Load experiment configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / config_file
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_constraints(chi_diag: np.ndarray, fitter: BaseAlternatingFitter) -> bool:
    """
    Validate that final chi satisfies all constraints.
    
    Returns
    -------
    bool
        True if all constraints satisfied
    """
    z_cons = fitter.orientation_via_axiality_constraint(chi_diag)
    quot_cons = fitter.orientation_rhombicity_constraint(chi_diag)
    trace_cons = fitter.trace_constraint(chi_diag)
    
    z_valid = all(c >= -1e-6 for c in z_cons)  # Allow small numerical errors
    quot_valid = all(c >= -1e-6 for c in quot_cons)
    trace_valid = all(abs(c) < 1e-6 for c in trace_cons)
    
    return z_valid and quot_valid and trace_valid


def compute_metrics(result: Dict, chi_true: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics for a single trial.
    
    Parameters
    ----------
    result : dict
        Result dictionary from fitter
    chi_true : np.ndarray
        Ground truth susceptibility tensor
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    chi_fitted = result['chi']
    
    metrics = {
        'final_loss': result['final_loss'],
        'n_iterations': result['n_iterations'],
        'converged': result['converged'],
        'chi_frobenius_error': np.linalg.norm(chi_fitted - chi_true, 'fro'),
        'chi_max_abs_error': np.max(np.abs(chi_fitted - chi_true))
    }
    
    return metrics


def run_chi_fit_experiment(config: Dict) -> Dict[str, Any]:
    """
    Run full experiment with multiple trials.
    
    Parameters
    ----------
    config : dict
        Experiment configuration
        
    Returns
    -------
    dict
        Experiment results including all trials
    """
    # Load data
    print("Loading experimental data...")
    hyperfines_dict, chi_true = load_hyperfines_and_susceptibility_tensor()
    delta_pc_dict = load_observed_pseudocontact_shift_data()
    
    # Extract parameters
    n_trials = config['optimization']['n_trials']
    max_iters = config['optimization']['max_iters']
    tol = config['optimization']['tolerance']
    optimizer = config['optimization']['optimizer']
    chi_iso = config['susceptibility']['chi_iso']
        
    OPTIMIZERS = {
        'alternating_optimization': AlternatingOptimizationFitter,
        'moment_matching': MomentMatchingFitter
    }
    optimizer_class = OPTIMIZERS.get(config['experiment']['name'])
    
    print(f"Running {n_trials} trials...")
    print(f"Optimizer: {optimizer}, Max iterations: {max_iters}, Tolerance: {tol}")
    
    results_list = []
    metrics_list = []
    
    for trial in range(n_trials):
        # Set seed
        if config['random_seed']['use_sequential']:
            seed = trial
        else:
            seed = config['random_seed']['base_seed'] + trial
        
        np.random.seed(seed)
        
        print(f"\nTrial {trial + 1}/{n_trials} (seed={seed})")
        
        # Generate random chi initialization for each trial
        from optimizers import generate_random_chi_diagonal
        chi_init = generate_random_chi_diagonal(chi_iso=chi_iso, seed=seed)
        
        # Common parameters for all fitters
        fitter_kwargs = {
            'hyperfines_dict': hyperfines_dict,
            'delta_pc_dict': delta_pc_dict,
            'chi_iso': chi_iso,
            'chi_init': chi_init,
            'max_iters': max_iters,
            'tol': tol,
            'optimizer': optimizer
        }
        
        # Add experiment-specific parameters
        if config['experiment']['name'] == 'moment_matching':
            # Use config parameters if available, otherwise use defaults
            moment_config = config.get('moment_matching', {})
            fitter_kwargs['max_moment'] = moment_config.get('max_moment', 12)
            fitter_kwargs['use_standardized'] = moment_config.get('use_standardized', True)
        
        # Create fitter and run
        fitter = optimizer_class(**fitter_kwargs)
        
        # Time the fitting process
        import time
        start_time = time.time()
        result = fitter.fit(verbose=config['output']['verbose'])
        elapsed_time = time.time() - start_time
        
        result["experiment_name"] = config['experiment']['name']
        result["elapsed_time"] = elapsed_time
        
        # Add frame_data from fitter to result (for animations)
        result['frame_data'] = fitter.frame_data if optimizer_class == AlternatingOptimizationFitter else fitter.chi_record
        
        # Validate constraints
        chi_diag = np.diag(result['chi'])
        constraints_valid = validate_constraints(chi_diag, fitter)
        
        if not constraints_valid:
            print(f"  WARNING: Trial {trial + 1}: Constraints violated!")
        
        # Compute metrics
        metrics = compute_metrics(result, chi_true)
        metrics['seed'] = seed
        metrics['elapsed_time'] = elapsed_time
        metrics['constraints_valid'] = constraints_valid
        
        print(f"  Final loss: {metrics['final_loss']:.6e}")
        print(f"  Chi error (Frobenius): {metrics['chi_frobenius_error']:.6e}")
        print(f"  Converged: {metrics['converged']}")
        
        results_list.append(result)
        metrics_list.append(metrics)
    
    # Find best result
    best_idx = np.argmin([m['final_loss'] for m in metrics_list])
    
    print("\n" + "="*60)
    print(f"BEST RESULT: Trial {best_idx + 1}")
    print("="*60)
    print(f"Final loss: {metrics_list[best_idx]['final_loss']:.6e}")
    print(f"Chi error: {metrics_list[best_idx]['chi_frobenius_error']:.6e}")
    print(f"Iterations: {metrics_list[best_idx]['n_iterations']}")
    
    # Aggregate results
    experiment_results = {        
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'data': {
            'chi_true': chi_true,
            'n_atoms': len(delta_pc_dict)
        },
        'trials': {
            'results': results_list,
            'metrics': metrics_list
        },
        'best_trial_idx': best_idx,
        'summary': {
            'mean_loss': np.mean([m['final_loss'] for m in metrics_list]),
            'std_loss': np.std([m['final_loss'] for m in metrics_list]),
            'min_loss': np.min([m['final_loss'] for m in metrics_list]),
            'mean_chi_error': np.mean([m['chi_frobenius_error'] for m in metrics_list]),
            'convergence_rate': np.mean([m['converged'] for m in metrics_list])
        }
    }
    
    return experiment_results


def save_results(results: Dict, config: Dict) -> Path:
    """
    Save experiment results to disk.
    
    Parameters
    ----------
    results : dict
        Experiment results
    config : dict
        Experiment configuration
        
    Returns
    -------
    Path
        Path to saved results file
    """
    if not config['output']['save_results']:
        print("Skipping results save (disabled in config)")
        return None
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    exp_name = config['experiment']['name']
    results_file = results_dir / f"{exp_name}_results.pkl"
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_file


def main():
    """Main entry point for running experiments."""
    print("="*60)
    print("ALTERNATING OPTIMIZATION EXPERIMENT")
    print("="*60)
    
    # Read in command line args
    parser = argparse.ArgumentParser(
        description='Run clustering experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='susc_fit.yaml',
        help='Config filename (e.g., base.yaml) or path to config file'
    )
    args = parser.parse_args()
    
    # Load config
    config_path = args.config + '.yaml' if not args.config.endswith('.yaml') else args.config
    config = load_config(config_path)
    
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    
    # Run experiment
    results = run_chi_fit_experiment(config)
    
    # Save results
    results_file = save_results(results, config)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total trials: {config['optimization']['n_trials']}")
    print(f"Best loss: {results['summary']['min_loss']:.6e}")
    print(f"Mean loss: {results['summary']['mean_loss']:.6e} ± {results['summary']['std_loss']:.6e}")
    print(f"Mean chi error: {results['summary']['mean_chi_error']:.6e}")
    print(f"Convergence rate: {results['summary']['convergence_rate']*100:.1f}%")
    
    if results_file:
        print(f"\nResults: {results_file}")


if __name__ == "__main__":
    main()
