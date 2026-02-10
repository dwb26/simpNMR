import pandas as pd
import numpy as np
from pathlib import Path

# Get the data directory path relative to this module
DATA_DIR = Path(__file__).parent.parent / 'data'

def _build_susceptibility_tensor(df) -> np.ndarray:
    """
    Construct a symmetric 3x3 tensor from 6 unique components.
        
    Returns:
        3x3 numpy array representing the symmetric tensor
    """
    xx = df["chi_xx (â„«^3)"]
    xy = df["chi_xy (â„«^3)"]
    xz = df["chi_xz (â„«^3)"]
    yy = df["chi_yy (â„«^3)"]
    yz = df["chi_yz (â„«^3)"]
    zz = df["chi_zz (â„«^3)"]
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]])


def _build_hyperfine_tensor(row) -> np.ndarray:
    """
    Extract hyperfine tensor components from a dataframe row.
    
    Args:
        row: Pandas Series containing Adip components
        
    Returns:
        3x3 numpy array representing the symmetric hyperfine tensor
    """
    xx=row['Adip_xx (ppm Å^-3)']
    xy=row['Adip_xy (ppm Å^-3)']
    xz=row['Adip_xz (ppm Å^-3)']
    yy=row['Adip_yy (ppm Å^-3)']
    yz=row['Adip_yz (ppm Å^-3)']
    zz=row['Adip_zz (ppm Å^-3)']
    return np.array([[xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]])


def _validate_traceless(tensor, tolerance=1e-10):
    """
    Validate that a tensor is traceless (trace ≈ 0).
    
    Args:
        tensor: numpy array to check
        tolerance: numerical tolerance for trace
        
    Raises:
        AssertionError if tensor is not traceless
    """
    trace = np.trace(tensor)
    assert np.isclose(trace, 0.0, atol=tolerance), \
        f"Tensor is not traceless! Trace = {trace:.6e}"


def _print_diagnostic_info(hyperfines_df, chi_tensor, hyperfine_tensors_dict):
    """Print diagnostic information about loaded tensors."""
    print("Hyperfines and shifts data")
    print("=" * 50)
    print(hyperfines_df.head())
    
    print("\n\nSusceptibility tensor χ:")
    print("=" * 50)
    print(chi_tensor)
    print(f"Trace: {np.trace(chi_tensor):.2e}")
    
    print("\n\nHyperfine tensors (A) for each atom:")
    print("=" * 50)
    for atom_label, tensor in hyperfine_tensors_dict.items():
        print(f"\n{atom_label}:")
        print(tensor)


def load_hyperfines_and_susceptibility_tensor(verbose=False) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load hyperfine tensors and susceptibility tensor from input data.
    
    Args:
        verbose: If True, print diagnostic information
        
    Returns:
        tuple: (hyperfine_tensors_dict, chi_tensor)
            - hyperfine_tensors_dict: Dictionary mapping atom labels to 3x3 hyperfine tensors
            - chi_tensor: 3x3 susceptibility tensor
    """
    # Load data
    hyperfines_df = pd.read_csv(DATA_DIR / 'hyperfines_and_shifts_298.00_K.csv')
    susceptibility_tensor_df = pd.read_excel(DATA_DIR / 'susceptibility_tensor.xlsx')
    
    # Read in susceptibility chi from file instead of hardcoding
    chi_tensor = _build_susceptibility_tensor(susceptibility_tensor_df.iloc[0])
    
    # Validate that susceptibility tensor is traceless
    _validate_traceless(chi_tensor)

    # Extract hyperfine tensors for each atom
    hyperfine_tensors_dict = {}
    for _, row in hyperfines_df.iterrows():
        atom_label = row['atom_label ()']
        hyperfine_tensor = _build_hyperfine_tensor(row)
        hyperfine_tensors_dict[atom_label] = hyperfine_tensor
    
    # Print diagnostic information if requested
    if verbose:
        _print_diagnostic_info(hyperfines_df, chi_tensor, hyperfine_tensors_dict)
    
    return hyperfine_tensors_dict, chi_tensor


def load_observed_pseudocontact_shift_data() -> dict[str, float]:
    """
    Load observed pseudocontact shift data from input file.
    
    Returns:
        dict: Dictionary mapping atom labels to pseudocontact shift values
    """
    # Read in the hyperfines and shifts file
    hyperfines_and_shifts_df = pd.read_csv(DATA_DIR / 'hyperfines_and_shifts_298.00_K.csv')
    
    delta_pc_dict = {}
    for _, row in hyperfines_and_shifts_df.iterrows():
        atom_label = row['atom_label ()']
        delta_pc = row['δ_pc (ppm)']
        delta_pc_dict[atom_label] = delta_pc
    
    return delta_pc_dict

# ht_dict, chi = load_hyperfines_and_susceptibility_tensor()