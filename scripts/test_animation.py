"""Quick test to verify animation functionality"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_handling import load_hyperfines_and_susceptibility_tensor, load_observed_pseudocontact_shift_data
from optimizers import AlternatingOptimisationFitter

print("Loading data...")
hyperfines_dict, chi_true = load_hyperfines_and_susceptibility_tensor()
delta_pc_dict = load_observed_pseudocontact_shift_data()

print("Running single trial...")
np.random.seed(6)  # Known good seed
fitter = AlternatingOptimisationFitter(
    hyperfines_dict=hyperfines_dict,
    delta_pc_dict=delta_pc_dict,
    chi_iso=0.18013,
    max_iters=10,
    tol=1e-8,
    optimizer='trust-constr'
)

result = fitter.fit(verbose=True)

print(f"\nResult keys: {result.keys()}")
print(f"Frame data available: {'frame_data' in result}")
if 'frame_data' not in result:
    print("Adding frame_data from fitter...")
    result['frame_data'] = fitter.frame_data
    print(f"Frame data length: {len(result['frame_data'])}")

# Test animation
if len(result['frame_data']) > 0:
    from analyze_results import create_animation
    print("\nCreating test animation...")
    create_animation(
        result['frame_data'],
        output_file='test_animation.mp4',
        fps=2
    )
    print("Animation created successfully!")
else:
    print("No frame data available")
