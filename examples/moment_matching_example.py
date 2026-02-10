"""
Example script showing how to use the moment matching animation.

Usage:
    python examples/moment_matching_example.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from data_handling import (
    load_hyperfines_and_susceptibility_tensor,
    load_observed_pseudocontact_shift_data
)
from optimizers import MomentMatchingFitter

# Import animation function
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from analyze_results import create_moment_matching_animation

print("="*60)
print("MOMENT MATCHING FITTER EXAMPLE")
print("="*60)

# Load data
print("\nLoading data...")
hyperfines_dict, chi_true = load_hyperfines_and_susceptibility_tensor()
delta_pc_dict = load_observed_pseudocontact_shift_data()

print(f"Number of atoms: {len(delta_pc_dict)}")
print(f"True chi tensor:\n{chi_true}")

# Run with centralized moments (to show the problem)
print("\n" + "-"*60)
print("Test 1: Centralized Moments (problematic)")
print("-"*60)
fitter_centralized = MomentMatchingFitter(
    hyperfines_dict=hyperfines_dict,
    delta_pc_dict=delta_pc_dict,
    max_moment=6,
    use_standardized=False,  # Use centralized moments
    chi_iso=0.18013,
    max_iters=50,
    optimizer='trust-constr'
)

result_centralized = fitter_centralized.fit(verbose=True)

print(f"\nFitted chi:\n{result_centralized['chi']}")
print(f"Chi error (Frobenius): {np.linalg.norm(result_centralized['chi'] - chi_true, 'fro'):.6e}")

# Create animation
print("\nCreating animation for centralized moments...")
create_moment_matching_animation(
    frame_data=result_centralized['frame_data'],
    output_file='results/moment_matching_centralized.mp4',
    fps=2,
    use_standardized=False
)

# Run with standardized moments (improved)
print("\n" + "-"*60)
print("Test 2: Standardized Moments (improved)")
print("-"*60)
fitter_standardized = MomentMatchingFitter(
    hyperfines_dict=hyperfines_dict,
    delta_pc_dict=delta_pc_dict,
    max_moment=6,
    use_standardized=True,  # Use standardized moments
    chi_iso=0.18013,
    max_iters=50,
    optimizer='trust-constr'
)

result_standardized = fitter_standardized.fit(verbose=True)

print(f"\nFitted chi:\n{result_standardized['chi']}")
print(f"Chi error (Frobenius): {np.linalg.norm(result_standardized['chi'] - chi_true, 'fro'):.6e}")

# Create animation
print("\nCreating animation for standardized moments...")
create_moment_matching_animation(
    frame_data=result_standardized['frame_data'],
    output_file='results/moment_matching_standardized.mp4',
    fps=2,
    use_standardized=True
)

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print("\nAnimations saved:")
print("  - results/moment_matching_centralized.mp4")
print("  - results/moment_matching_standardized.mp4")
print("\nCompare the two to see the difference in convergence behavior!")
