"""
Noise sensitivity experiment for chi initialization.

This script systematically tests how robust the alternating optimization
algorithm is to noise in the initial guess for the susceptibility tensor.

Usage:
    python run_noise_experiment.py [--config CONFIG_FILE]
"""

import numpy as np
import pickle
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_handling import (
    load_hyperfines_and_susceptibility_tensor,
    load_observed_pseudocontact_shift_data
)
from optimizers import AlternatingOptimisationFitter


def setup_logging(log_file: Path) -> None:
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_file: str = "experiments/config_noise_sensitivity.yaml") -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def add_noise_to_chi(
    chi_true: np.ndarray, 
    noise_level: float, 
    noise_type: str = 'relative',
    enforce_traceless: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Add noise to susceptibility tensor.
    
    Parameters
    ----------
    chi_true : np.ndarray
        Ground truth chi tensor (3x3)
    noise_level : float
        Magnitude of noise (interpretation depends on noise_type)
    noise_type : str
        'relative': noise_level is fraction of true values (e.g., 0.01 = 1%)
        'absolute': noise_level is absolute magnitude
    enforce_traceless : bool
        If True, project back to traceless manifold after adding noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Noisy chi tensor (3x3)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random noise matrix
    noise = np.random.randn(3, 3)
    
    # Make noise symmetric (chi should be symmetric)
    noise = (noise + noise.T) / 2
    
    if noise_type == 'relative':
        # Scale noise relative to magnitude of true chi
        chi_noisy = chi_true + noise_level * np.abs(chi_true) * noise
    else:  # absolute
        chi_noisy = chi_true + noise_level * noise
    
    if enforce_traceless:
        # Project back to traceless manifold
        trace_chi = np.trace(chi_noisy)
        chi_noisy = chi_noisy - (trace_chi / 3) * np.eye(3)
    
    return chi_noisy


def check_assignment_correct(
    result: Dict,
    chi_true: np.ndarray,
    assignment_tolerance: float = 1e-4,
    chi_tolerance: float = 1e-2
) -> Tuple[bool, bool]:
    """
    Check if the result recovered correct assignment and chi.
    
    Parameters
    ----------
    result : dict
        Optimization result
    chi_true : np.ndarray
        Ground truth chi tensor
    assignment_tolerance : float
        Tolerance for considering assignment correct (based on loss)
    chi_tolerance : float
        Tolerance for chi error (Frobenius norm)
        
    Returns
    -------
    Tuple[bool, bool]
        (assignment_correct, chi_correct)
    """
    # Assignment is correct if loss is very small
    assignment_correct = result['final_loss'] < assignment_tolerance
    
    # Chi is correct if Frobenius error is small
    chi_error = np.linalg.norm(result['chi'] - chi_true, 'fro')
    chi_correct = chi_error < chi_tolerance
    
    return assignment_correct, chi_correct


def run_noise_sweep(config: Dict) -> Dict:
    """
    Run noise sensitivity sweep experiment.
    
    Parameters
    ----------
    config : dict
        Experiment configuration
        
    Returns
    -------
    dict
        Results for all noise levels
    """
    # Load data
    logging.info("Loading experimental data...")
    hyperfines_dict, chi_true = load_hyperfines_and_susceptibility_tensor()
    delta_pc_dict = load_observed_pseudocontact_shift_data()
    
    noise_levels = config['noise']['levels']
    trials_per_level = config['noise']['trials_per_level']
    noise_type = config['noise']['noise_type']
    enforce_traceless = config['noise']['enforce_traceless']
    
    logging.info(f"Running noise sensitivity sweep...")
    logging.info(f"Noise levels: {noise_levels}")
    logging.info(f"Trials per level: {trials_per_level}")
    logging.info(f"Noise type: {noise_type}")
    
    # Storage for results
    sweep_results = {
        'noise_levels': noise_levels,
        'trials_per_level': trials_per_level,
        'level_results': []
    }
    
    for noise_idx, noise_level in enumerate(noise_levels):
        logging.info(f"\n{'='*60}")
        logging.info(f"NOISE LEVEL: {noise_level*100:.1f}%")
        logging.info(f"{'='*60}")
        
        level_trials = []
        n_success = 0
        
        for trial in range(trials_per_level):
            # Incorporate noise level index to ensure different seeds across noise levels
            base_seed = config['random_seed']['base_seed'] if not config['random_seed']['use_sequential'] else 0
            seed = base_seed + noise_idx * 1000 + trial
            
            # Generate noisy chi_init
            chi_init = add_noise_to_chi(
                chi_true, 
                noise_level, 
                noise_type=noise_type,
                enforce_traceless=enforce_traceless,
                seed=seed * 1000  # Different seed for noise generation
            )
            
            # Run optimization with noisy initialization
            np.random.seed(seed)  # For assignment initialization
            fitter = AlternatingOptimisationFitter(
                hyperfines_dict=hyperfines_dict,
                delta_pc_dict=delta_pc_dict,
                chi_iso=config['susceptibility']['chi_iso'],
                chi_init=chi_init,
                max_iters=config['optimization']['max_iters'],
                tol=config['optimization']['tolerance'],
                optimizer=config['optimization']['optimizer']
            )
            
            result = fitter.fit(verbose=False)
            
            # Check success
            assignment_correct, chi_correct = check_assignment_correct(result, chi_true)
            if assignment_correct:
                n_success += 1
            
            # Store trial result
            trial_result = {
                'seed': seed,
                'chi_init': chi_init,
                'chi': result['chi'],
                'final_loss': result['final_loss'],
                'chi_error': np.linalg.norm(result['chi'] - chi_true, 'fro'),
                'n_iterations': result['n_iterations'],
                'converged': result['converged'],
                'assignment_correct': assignment_correct,
                'chi_correct': chi_correct
            }
            
            if config['output'].get('save_individual_trials', False):
                trial_result['full_result'] = result
            
            level_trials.append(trial_result)
            
            if config['output']['verbose']:
                logging.info(f"  Trial {trial+1}/{trials_per_level}: "
                           f"Loss={result['final_loss']:.2e}, "
                           f"χ_error={trial_result['chi_error']:.2e}")
        
        # Find best trial (lowest loss)
        best_trial_idx = np.argmin([t['final_loss'] for t in level_trials])
        best_trial = level_trials[best_trial_idx]
        
        success_rate = n_success / trials_per_level
        
        logging.info(f"\nNoise level {noise_level*100:.1f}% summary:")
        logging.info(f"  Best trial: {best_trial_idx+1}/{trials_per_level}")
        logging.info(f"  Best loss: {best_trial['final_loss']:.2e}")
        logging.info(f"  Best χ error: {best_trial['chi_error']:.2e}")
        logging.info(f"  Success rate: {success_rate*100:.1f}% ({n_success}/{trials_per_level})")
        
        level_summary = {
            'noise_level': noise_level,
            'trials': level_trials,
            'best_trial_idx': best_trial_idx,
            'best_chi_init': best_trial['chi_init'],
            'best_chi': best_trial['chi'],
            'best_loss': best_trial['final_loss'],
            'best_chi_error': best_trial['chi_error'],
            'best_n_iterations': best_trial['n_iterations'],
            'success_rate': success_rate,
            'n_success': n_success,
            'mean_loss': np.mean([t['final_loss'] for t in level_trials]),
            'mean_chi_error': np.mean([t['chi_error'] for t in level_trials]),
            'std_chi_error': np.std([t['chi_error'] for t in level_trials])
        }
        
        sweep_results['level_results'].append(level_summary)
    
    # Add metadata
    sweep_results['config'] = config
    sweep_results['timestamp'] = datetime.now().isoformat()
    sweep_results['chi_true'] = chi_true
    
    return sweep_results


def save_results(results: Dict, config: Dict) -> Path:
    """Save experiment results to disk."""
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    results_file = results_dir / f"results_{exp_name}_{timestamp}.pkl"
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"\nResults saved to: {results_file}")
    return results_file


def print_summary(results: Dict):
    """Print summary of noise sensitivity results."""
    logging.info("\n" + "="*60)
    logging.info("NOISE SENSITIVITY SUMMARY")
    logging.info("="*60)
    
    logging.info(f"\n{'Noise Level':<15} {'Best Loss':<15} {'Best χ Error':<15} {'Mean χ Error':<15}")
    logging.info("-"*60)
    
    for level_result in results['level_results']:
        noise_pct = level_result['noise_level'] * 100
        best_loss = level_result['best_loss']
        best_chi_error = level_result['best_chi_error']
        mean_error = level_result['mean_chi_error']
        
        logging.info(f"{noise_pct:>6.1f}%        {best_loss:>10.2e}        {best_chi_error:>10.2e}        {mean_error:>10.2e}")
    
    # Find where best error degrades significantly
    first_best_error = results['level_results'][0]['best_chi_error']
    for level_result in results['level_results']:
        if level_result['best_chi_error'] > first_best_error * 10:  # 10x degradation
            logging.info(f"\n⚠️  Best χ error degrades >10x at {level_result['noise_level']*100:.1f}% noise")
            break
    else:
        max_noise = results['level_results'][-1]['noise_level'] * 100
        logging.info(f"\n✓  Best χ error remains within 10x of baseline up to {max_noise:.1f}% noise")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run noise sensitivity experiment")
    parser.add_argument('--config', default='experiments/config_noise_sensitivity.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"experiments/results/noise_experiment_{timestamp}.log")
    setup_logging(log_file)
    
    logging.info("="*60)
    logging.info("NOISE SENSITIVITY EXPERIMENT")
    logging.info("="*60)
    
    # Load config
    config = load_config(args.config)
    logging.info(f"\nExperiment: {config['experiment']['name']}")
    logging.info(f"Description: {config['experiment']['description']}")
    
    # Run experiment
    results = run_noise_sweep(config)
    
    # Save
    results_file = save_results(results, config)
    
    # Summary
    print_summary(results)
    
    logging.info(f"\nResults: {results_file}")
    logging.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
