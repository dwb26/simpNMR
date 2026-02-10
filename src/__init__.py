"""
Core modules for paramagnetic NMR optimization.

This package contains:
- optimizers: Alternating optimization algorithms
- data_handling: Data loading and preprocessing utilities
- Gaussian_fit: Gaussian fitting functions
"""

__version__ = "0.1.0"

from .optimizers import (
    BaseAlternatingFitter,
    AlternatingOptimisationFitter,
    MomentMatchingFitter
)

from .data_handling import (
    load_hyperfines_and_susceptibility_tensor,
    load_observed_pseudocontact_shift_data
)

__all__ = [
    'BaseAlternatingFitter',
    'AlternatingOptimisationFitter',
    'AxialRhombicityFitter',
    'MomentMatchingFitter',
    'load_hyperfines_and_susceptibility_tensor',
    'load_observed_pseudocontact_shift_data',
]
