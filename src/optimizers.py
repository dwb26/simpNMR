"""
Optimization algorithms for paramagnetic NMR assignment and susceptibility fitting.

This module contains implementations of optimization methods for jointly solving
the assignment problem (matching hyperfine tensors to atoms) and fitting the
magnetic susceptibility tensor.
"""

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from typing import Dict, Optional, List
from numpy.typing import NDArray


def generate_random_chi_diagonal(chi_iso: float = 0.18013, seed: Optional[int] = None) -> NDArray:
    """
    Generate a random diagonal chi tensor satisfying physical constraints.
    
    Constraints satisfied:
    1. Traceless: chi_x + chi_y + chi_z = 0
    2. Axiality: -chi_iso < chi_z < 2*chi_iso
    3. Rhombicity: 0 < (chi_x - chi_y)/chi_z < 1
    
    Parameters
    ----------
    chi_iso : float
        Isotropic susceptibility (used for bounds)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    NDArray
        3x3 diagonal chi tensor satisfying constraints
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample chi_z from valid range: -chi_iso < chi_z < 2*chi_iso
    # Avoid chi_z = 0 for rhombicity constraint
    chi_z = np.random.uniform(-chi_iso * 0.95, 2 * chi_iso * 0.95)
    if abs(chi_z) < 0.01 * chi_iso:  # Avoid division by zero
        chi_z = 0.01 * chi_iso if chi_z >= 0 else -0.01 * chi_iso
    
    # Sample rhombicity: 0 < (chi_x - chi_y)/chi_z < 1
    rho = np.random.uniform(0.05, 0.95)  # Avoid boundary
    
    # Solve for chi_x and chi_y
    # We have: chi_x + chi_y = -chi_z (traceless)
    # And: chi_x - chi_y = rho * chi_z (rhombicity)
    # Therefore: chi_x = (-chi_z + rho * chi_z) / 2
    #            chi_y = (-chi_z - rho * chi_z) / 2
    
    chi_x = 0.5 * (-chi_z + rho * chi_z)
    chi_y = 0.5 * (-chi_z - rho * chi_z)
    
    # Create diagonal tensor
    chi = np.diag([chi_x, chi_y, chi_z])
    
    # Verify constraints (for debugging)
    assert abs(chi_x + chi_y + chi_z) < 1e-10, "Traceless constraint violated"
    assert -chi_iso < chi_z < 2 * chi_iso, "Axiality constraint violated"
    rhombicity = (chi_x - chi_y) / chi_z if chi_z != 0 else 0
    assert 0 < rhombicity < 1, f"Rhombicity constraint violated: {rhombicity}"
    
    return chi


class BaseAlternatingFitter:
    """
    Base class for alternating optimization methods.
    
    This abstract base class provides the common framework for alternating between:
    1. Fixing the assignment and optimizing the susceptibility tensor χ
    2. Fixing χ and optimizing the assignment using the Hungarian algorithm
    
    Subclasses must implement the `fit_chi()` method to define how χ is
    parameterized and optimized.
    
    Attributes
    ----------
    atom_labels : list[str]
        Labels of atoms in the molecule
    N_atoms : int
        Number of atoms
    observed_shifts : NDArray
        Observed pseudocontact shifts (ppm)
    hyperfine_tensors_original : list[NDArray]
        Original hyperfine coupling tensors (3x3 arrays)
    current_permutation : NDArray
        Current assignment permutation
    hyperfines_dict : dict
        Current assignment of hyperfines to atoms
    chi_iso : float
        Isotropic susceptibility (fixed)
    chi : NDArray
        Current susceptibility tensor (3x3 array)
    max_iters : int
        Maximum alternating iterations
    tol : float
        Convergence tolerance
    loss_history : list[float]
        Loss at each iteration
    assignment_history : list[NDArray]
        Assignment at each iteration
    chi_history : list[NDArray]
        Chi tensor at each iteration
    frame_data : list[dict]
        Detailed frame-by-frame data for visualization
    """
    
    def __init__(
        self, 
        hyperfines_dict: Dict[str, NDArray], 
        delta_pc_dict: Dict[str, float], 
        chi_iso: float = 0.18013, 
        chi_init: Optional[NDArray] = None, 
        max_iters: int = 50, 
        tol: float = 1e-6,
        optimizer: str = 'trust-constr'
    ):
        """
        Initialize the alternating optimization fitter.
        
        Parameters
        ----------
        hyperfines_dict : dict[str, NDArray]
            Dictionary mapping atom labels to hyperfine tensors (3x3 arrays)
        delta_pc_dict : dict[str, float]
            Dictionary mapping atom labels to observed pseudocontact shifts (ppm)
        chi_iso : float, default=0.18013
            Isotropic susceptibility component (fixed)
        chi_init : NDArray, optional
            Initial guess for susceptibility tensor (3x3). If None, generates random valid tensor.
        max_iters : int, default=50
            Maximum number of alternating iterations
        tol : float, default=1e-6
            Convergence tolerance for loss change
        optimizer : str, default='trust-constr'
            Scipy optimizer to use ('trust-constr', 'SLSQP', etc.)
        """
        self.atom_labels = list(delta_pc_dict.keys())
        self.N_atoms = len(self.atom_labels)
        self.observed_shifts = np.array([delta_pc_dict[label] for label in self.atom_labels])
        
        # Store original hyperfines as list (we'll permute these)
        self.hyperfine_tensors_original = [hyperfines_dict[label] for label in self.atom_labels]
        
        # Initialize with random permutation to model unknown assignment scenario
        self.current_permutation = np.random.permutation(len(self.atom_labels))
        self.hyperfines_dict = self._build_assignment(self.current_permutation)
        
        self.chi_iso = chi_iso if chi_iso is not None else 0.0
        
        # Initialize chi tensor with random valid values if not provided
        if chi_init is not None:
            self.chi = chi_init.copy()
        else:
            self.chi = generate_random_chi_diagonal(chi_iso=self.chi_iso)
        
        self.max_iters = max_iters
        self.tol = tol
        self.optimizer = optimizer
        
        # Track convergence
        self.loss_history: List[float] = []
        self.assignment_history: List[NDArray] = []
        self.chi_history: List[NDArray] = []
        self.frame_data: List[dict] = []
        self.chi_record: List[NDArray] = []
    
    
    def forward_model(self, chi: NDArray) -> NDArray:
        """
        Compute predicted pseudocontact shifts.
        
        This method must be implemented by subclasses to define their
        specific forward model equation.
        
        Parameters
        ----------
        chi : NDArray
            Susceptibility tensor (3x3) or other representation
            
        Returns
        -------
        NDArray
            Predicted pseudocontact shifts for all atoms
        """
        raise NotImplementedError("Subclasses must implement forward_model()")
    
    
    def fit_chi(self) -> None:
        """
        Step 1: Fix assignment and optimize susceptibility tensor χ.
        
        This method must be implemented by subclasses to define the
        parameterization and constraints for χ optimization.
        
        Subclasses should:
        1. Define parameters to optimize (e.g., [chi_x, chi_y, chi_z] or [chi_ax, chi_rho, euler_angles])
        2. Optimize those parameters
        3. Convert to 3x3 tensor and update self.chi
        """
        raise NotImplementedError("Subclasses must implement fit_chi()")
    
    
    def optimize_assignment(self) -> NDArray:
        """
        Step 2: Fix χ and optimise assignment using Hungarian algorithm.
        
        Solves the linear assignment problem to find the permutation of
        hyperfine tensors that minimizes the total squared error.
        
        Returns
        -------
        NDArray
            Optimal assignment permutation indices
        """
        cost_matrix = np.zeros((self.N_atoms, self.N_atoms))
        
        # Build cost matrix: C[i,j] = cost of assigning hyperfine j to atom i
        for i in range(self.N_atoms):
            for j in range(self.N_atoms):
                predicted_shift = (1/3) * np.trace(
                    self.hyperfine_tensors_original[j] @ self.chi
                )
                cost_matrix[i, j] = (self.observed_shifts[i] - predicted_shift)**2
        
        # Solve linear assignment problem
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update permutation and assignment
        self.current_permutation = col_ind
        self.hyperfines_dict = self._build_assignment(col_ind)
        
        return col_ind
    
    
    def _compute_current_loss(self) -> float:
        """
        Compute loss for current chi and assignment.
        
        Returns
        -------
        float
            Mean squared error
        """
        y_hat = self.forward_model(self.chi)
        return np.mean((y_hat - self.observed_shifts)**2)
    
    
    def fit(self, verbose: bool = False) -> dict:
        """
        Run alternating optimisation until convergence.
        
        Parameters
        ----------
        verbose : bool, default=False
            If True, print iteration information
        
        Returns
        -------
        dict
            Results dictionary with keys:
            - 'chi': Fitted susceptibility tensor (3x3)
            - 'assignment': Final permutation indices
            - 'assignment_dict': Mapping of atom labels to assigned atoms
            - 'loss_history': Loss at each iteration
            - 'chi_history': Chi tensors at each iteration
            - 'converged': Whether algorithm converged within tolerance
            - 'final_loss': Final loss value
            - 'n_iterations': Number of iterations performed
        """
        # Store initial state
        predicted_shifts_init = self.forward_model(self.chi)
        initial_loss = self._compute_current_loss()
        self.frame_data.append({
            'iteration': 0,
            'step': 'initial',
            'chi': self.chi.copy(),
            'assignment': self.current_permutation.copy(),
            'predicted': predicted_shifts_init.copy(),
            'observed': self.observed_shifts.copy(),
            'loss': initial_loss
        })
        
        if verbose:
            print(f"Initial loss: {initial_loss:.6e}")
        
        for iteration in range(self.max_iters):
            
            # Step 1: Optimize χ given assignment
            self.fit_chi()
            loss_after_chi = self._compute_current_loss()
            
            # Store frame after chi optimization
            predicted_shifts_chi = self.forward_model(self.chi)
            self.frame_data.append({
                'iteration': iteration + 1,
                'step': 'chi_optimized',
                'chi': self.chi.copy(),
                'assignment': self.current_permutation.copy(),
                'predicted': predicted_shifts_chi.copy(),
                'observed': self.observed_shifts.copy(),
                'loss': loss_after_chi
            })
            
            # Step 2: Optimize assignment given χ
            new_assignment = self.optimize_assignment()
            loss_after_assignment = self._compute_current_loss()
            
            # Store frame after assignment optimization
            predicted_shifts_assign = self.forward_model(self.chi)
            self.frame_data.append({
                'iteration': iteration + 1,
                'step': 'assignment_optimized',
                'chi': self.chi.copy(),
                'assignment': new_assignment.copy(),
                'predicted': predicted_shifts_assign.copy(),
                'observed': self.observed_shifts.copy(),
                'loss': loss_after_assignment
            })
            
            # Track history
            self.loss_history.append(loss_after_assignment)
            self.assignment_history.append(new_assignment.copy())
            self.chi_history.append(self.chi.copy())
            
            if verbose:
                print(f"Iteration {iteration+1}: Loss = {loss_after_assignment:.6e}")
            
            # Check convergence
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - self.loss_history[-1])
                if loss_change < self.tol:
                    if verbose:
                        print(f"Converged after {iteration+1} iterations")
                    break
        
        # Build final assignment dict for interpretation
        final_assignment_dict = {
            self.atom_labels[i]: self.atom_labels[self.current_permutation[i]]
            for i in range(self.N_atoms)
        }
        
        return {
            'chi': self.chi,
            'assignment': self.current_permutation,
            'assignment_dict': final_assignment_dict,
            'loss_history': self.loss_history,
            'chi_history': self.chi_history,
            'converged': iteration < self.max_iters - 1,
            'final_loss': self.loss_history[-1],
            'n_iterations': len(self.loss_history)
        }
    
    
    def _build_assignment(self, permutation: NDArray) -> Dict[str, NDArray]:
        """
        Build hyperfines_dict from permutation indices.
        
        Parameters
        ----------
        permutation : NDArray
            Permutation indices
            
        Returns
        -------
        dict
            Dictionary mapping atom labels to assigned hyperfine tensors
        """
        return {
            self.atom_labels[i]: self.hyperfine_tensors_original[permutation[i]]
            for i in range(len(self.atom_labels))
        }


class AlternatingOptimizationFitter(BaseAlternatingFitter):
    """
    Diagonal chi tensor optimization with axiality/rhombicity constraints.
    
    This implementation parameterizes χ as a diagonal tensor and optimizes
    the three diagonal components subject to:
    - Traceless constraint: chi_x + chi_y + chi_z = 0
    - Axiality bounds: -chi_iso < chi_z < 2*chi_iso
    - Rhombicity bounds: 0 < (chi_x - chi_y)/chi_z < 1
    
    Forward model: δ_pc = (1/3) * Tr(A @ χ)
    
    Inherits the alternating optimization framework from BaseAlternatingFitter.
    """    
    
    def forward_model(self, chi: NDArray) -> NDArray:
        """
        Compute predicted pseudocontact shifts using diagonal chi tensor.
        
        Uses the formula: δ_pc = (1/3) * Tr(A @ χ)
        
        Parameters
        ----------
        chi : NDArray
            Susceptibility tensor (3x3 diagonal)
            
        Returns
        -------
        NDArray
            Predicted pseudocontact shifts for all atoms
        """
        y_hat = np.array([
            (1/3) * np.trace(self.hyperfines_dict[label] @ chi)
            for label in self.atom_labels
        ])
        return y_hat
    
    
    def loss_fn(self, chi_diag: NDArray) -> float:
        """
        Loss function: mean squared error between predicted and observed shifts.
        
        Parameters
        ----------
        chi_diag : NDArray
            Diagonal elements of chi tensor [chi_x, chi_y, chi_z]
            
        Returns
        -------
        float
            Mean squared error
        """
        chi_hat = np.diag(chi_diag)
        y_hat = self.forward_model(chi_hat)
        loss_val = np.mean((y_hat - self.observed_shifts)**2)
        return loss_val
    
    
    def orientation_via_axiality_constraint(self, chi_diag: NDArray) -> List[float]:
        """
        Inequality constraint: -chi_iso < chi_z < 2*chi_iso
        
        Returns positive values when constraint is satisfied.
        """
        chi_z = chi_diag[2]
        return [chi_z + self.chi_iso, 2*self.chi_iso - chi_z]
    
    
    def orientation_rhombicity_constraint(self, chi_diag: NDArray) -> List[float]:
        """
        Inequality constraint: 0 < (chi_x - chi_y)/chi_z < 1 (rhombicity)
        
        Returns positive values when constraint is satisfied.
        """
        chi_x, chi_y, chi_z = chi_diag[0], chi_diag[1], chi_diag[2]
        ratio = (chi_x - chi_y) / chi_z if chi_z != 0 else 0.0
        return [ratio, 1 - ratio]
    
    
    def trace_constraint(self, chi_diag: NDArray) -> List[float]:
        """
        Equality constraint: Tr(chi) = 0 (traceless tensor)
        
        Returns zero when constraint is satisfied.
        """
        return [chi_diag[0] + chi_diag[1] + chi_diag[2]]
    
    
    def fit_chi(self) -> None:
        """
        Optimize diagonal chi tensor with axiality/rhombicity constraints.
        
        Minimizes the loss function subject to:
        - Traceless constraint: chi_x + chi_y + chi_z = 0
        - Axiality bounds: -chi_iso < chi_z < 2*chi_iso
        - Rhombicity bounds: 0 < (chi_x - chi_y)/chi_z < 1
        """
        constraints = [
            {'type': 'eq', 'fun': self.trace_constraint},
            {'type': 'ineq', 'fun': self.orientation_via_axiality_constraint},
            {'type': 'ineq', 'fun': self.orientation_rhombicity_constraint}
        ]
        
        def callback(chi_k, _):
            self.chi_record.append({
                'step': 'chi_optimization',
                'chi': np.diag(chi_k.copy()),
                'predicted_shifts': self.forward_model(np.diag(chi_k.copy())),
                'observed': self.observed_shifts.copy(),
                'loss': self.loss_fn(chi_k)
            })
        
        res = minimize(
            self.loss_fn, 
            x0=np.diag(self.chi),
            method=self.optimizer, 
            constraints=constraints,
            callback=callback
        )
        self.converged = res.success
        
        self.chi = np.diag(res.x)
        
        
class MomentMatchingFitter(AlternatingOptimizationFitter):
    """
    Fits for the susceptibility tensor by fitting for the probabilistic moments of the pseduocontact shifts.
    
    Parameters
    ----------
    hyperfines_dict : Dict[str, NDArray]
        Dictionary mapping atom labels to hyperfine tensors (3x3 arrays)
    delta_pc_dict : Dict[str, float]
        Dictionary mapping atom labels to observed pseudocontact shifts (ppm)
    max_moment : int, default=6
        Maximum moment to compute and fit
    """
    
    def __init__(
        self,
        hyperfines_dict: Dict[str, NDArray],
        delta_pc_dict: Dict[str, float],
        max_moment: int = 6,
        use_standardized: bool = False,
        **kwargs
    ):
        # Store true hyperfines dict BEFORE parent __init__ permutes it
        # We'll use this after fitting chi to compute predictions with correct assignment
        self.true_hyperfines_dict = hyperfines_dict.copy()
        
        super().__init__(hyperfines_dict, delta_pc_dict, **kwargs)
        self.max_moment = max_moment
        self.use_standardized = use_standardized
        
        # Compute moments using selected method
        if use_standardized:
            self.observed_moments = self.compute_standardized_moments(self.observed_shifts)
        else:
            self.observed_moments = self.compute_shift_moments(self.observed_shifts)
    
    
    def compute_shift_moments(self, deltas) -> NDArray:
        """
        Compute centralized moments of the pseudocontact shifts.
        
        WARNING: Higher moments (M5, M6) have huge magnitudes which can dominate
        the loss function. Use compute_standardized_moments() for better numerical behavior.
        
        Computes:
        - M1 = mean (raw moment)
        - M2 = sample variance (central moment with Bessel correction)
        - M3-M6 = central moments (population statistics)
        
        Parameters
        ----------
        deltas : NDArray
            Observed pseudocontact shifts
            
        Returns
        -------
        NDArray
            Array of moments [mean, variance, M3, M4, M5, M6]
        """
        moment_1_delta = np.mean(deltas)
        N = len(deltas)
        
        moments = np.zeros(self.max_moment)
        for i in range(1, self.max_moment + 1):
            if i == 1:
                moments[i - 1] = moment_1_delta
            elif i == 2:
                moments[i - 1] = (1 / (N - 1)) * np.sum((deltas - moment_1_delta)**2)
            else:
                moments[i - 1] = (1 / N) * np.sum((deltas - moment_1_delta)**i)
        
        return moments
    
    
    def compute_standardized_moments(self, deltas) -> NDArray:
        """
        Compute standardized moments of the pseudocontact shifts.
        
        Uses standardization to avoid numerical issues with large magnitude differences.
        All higher moments are dimensionless and O(1) in magnitude.
        
        Computes:
        - M1 = mean
        - M2 = standard deviation (not variance, to keep scale similar to M1)
        - M3 = standardized skewness (dimensionless)
        - M4 = standardized excess kurtosis (dimensionless)
        - M5, M6 = higher standardized moments (dimensionless)
        
        Parameters
        ----------
        deltas : NDArray
            Observed pseudocontact shifts
            
        Returns
        -------
        NDArray
            Array of standardized moments [mean, std, skew, kurtosis, ...]
        """
        N = len(deltas)
        mean = np.mean(deltas)
        std = np.std(deltas, ddof=1)  # Sample standard deviation
        
        # Avoid division by zero
        std = 1e-10 if std < 1e-10 else std
        
        moments = np.zeros(self.max_moment)
        moments[0] = mean
        moments[1] = std
        
        # Compute standardized higher moments
        centered = (deltas - mean) / std
        for k in range(3, self.max_moment + 1):
            moments[k - 1] = np.mean(centered ** k)
        
        return moments
        
    
    def loss_fn(self, chi_diag: NDArray) -> float:
        """
        Weighted loss function: mean squared error between predicted and observed moments.
        
        Uses optional weighting to balance contribution of different moments.
        
        Parameters
        ----------
        chi_diag : NDArray
            Diagonal elements of chi tensor [chi_x, chi_y, chi_z]
            
        Returns
        -------
        float
            Weighted mean squared error of moments
        """
        chi_hat = np.diag(chi_diag)
        predicted_moments = self.forward_model(chi_hat)
        
        # Option 1: Equal weighting after standardization
        loss_val = np.mean((predicted_moments - self.observed_moments)**2)
        
        # Option 2: Inverse variance weighting (uncomment to use)
        # weights = np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25])[:self.max_moment]
        # loss_val = np.mean(weights * (predicted_moments - self.observed_moments)**2)
        
        return loss_val
    
    
    def forward_model(self, chi: NDArray) -> NDArray:
        """
        Compute predicted pseudocontact shifts using moment-matching parameterization.
        
        Parameters
        ----------
        chi : NDArray
            Susceptibility tensor (3x3 diagonal)
            
        Returns
        -------
        NDArray
            Predicted pseudocontact shift moments for all atoms
        """
        
        # Compute the empirical moments of the predicted shifts
        delta_hat = np.array([
            (1/3) * np.trace(self.hyperfines_dict[label] @ chi)
            for label in self.atom_labels
        ])
        
        # Use the same moment computation method as initialization
        if self.use_standardized:
            return self.compute_standardized_moments(delta_hat)
        else:
            return self.compute_shift_moments(delta_hat)
    
    
    def _compute_current_loss(self) -> float:
        """
        Compute loss for current chi and assignment.
        
        Returns
        -------
        float
            Mean squared error
        """
        y_hat = self.forward_model(self.chi)
        return np.mean((y_hat - self.observed_moments)**2)
    
    
    def fit_chi(self) -> None:
        """
        Override parent to record both shifts and moments during optimization.
        """
        constraints = [
            {'type': 'eq', 'fun': self.trace_constraint},
            {'type': 'ineq', 'fun': self.orientation_via_axiality_constraint},
            {'type': 'ineq', 'fun': self.orientation_rhombicity_constraint}
        ]
        
        def callback(chi_k, _):
            chi_current = np.diag(chi_k.copy())
            
            # Compute predicted shifts using TRUE assignment (for left animation)
            # This shows how well the fitted chi matches the observations when assignment is correct
            predicted_shifts = np.array([
                (1/3) * np.trace(self.true_hyperfines_dict[label] @ chi_current)
                for label in self.atom_labels
            ])
            
            # Compute predicted moments using the permuted hyperfines (for fitting)
            # The optimization still fits based on moment matching without knowing assignment
            predicted_shifts_permuted = np.array([
                (1/3) * np.trace(self.hyperfines_dict[label] @ chi_current)
                for label in self.atom_labels
            ])
            
            if self.use_standardized:
                predicted_moments = self.compute_standardized_moments(predicted_shifts_permuted)
            else:
                predicted_moments = self.compute_shift_moments(predicted_shifts_permuted)
            
            # Compute loss for this iteration
            loss_value = self.loss_fn(chi_k)
            
            self.chi_record.append({
                'step': 'chi_optimization',
                'chi': chi_current,
                'predicted_shifts': predicted_shifts,  # Using TRUE assignment for visualization
                'observed_shifts': self.observed_shifts.copy(),
                'predicted_moments': predicted_moments,
                'observed_moments': self.observed_moments.copy(),
                'loss': loss_value
            })
        
        res = minimize(
            self.loss_fn, 
            x0=np.diag(self.chi),
            method=self.optimizer, 
            constraints=constraints,
            callback=callback
        )
        self.converged = res.success
        self.chi = np.diag(res.x)
    
    
    def fit(self, verbose: bool = False) -> dict:
        """
        Fitting but with no alternating assignment step

        Args:
            verbose (bool, optional): Print progress information. Defaults to False.

        Returns:
            dict: Results dictionary
        """
        # Compute initial predictions using TRUE assignment for visualization
        predicted_shifts_init = np.array([
            (1/3) * np.trace(self.true_hyperfines_dict[label] @ self.chi)
            for label in self.atom_labels
        ])
        
        # Compute moments using permuted hyperfines (for loss calculation)
        predicted_moments_init = self.forward_model(self.chi)
        initial_loss = self._compute_current_loss()
        
        # Store initial state
        self.loss_history = []
        self.loss_history.append(initial_loss)
        self.chi_record.append({
            'step': 'initial',
            'chi': self.chi.copy(),
            'predicted_shifts': predicted_shifts_init.copy(),  # Using TRUE assignment
            'observed_shifts': self.observed_shifts.copy(),
            'predicted_moments': predicted_moments_init.copy(),
            'observed_moments': self.observed_moments.copy(),
            'loss': initial_loss,
            'loss_history': self.loss_history.copy()
        })
        
        if verbose:
            print(f"Initial loss: {initial_loss:.6e}")
            print(f"Observed moments: {self.observed_moments}")
            print(f"Predicted moments: {predicted_moments_init}")
        
        # Optimize χ
        self.fit_chi()
        loss_after_chi = self._compute_current_loss()
        
        if verbose:
            print(f"Final loss: {loss_after_chi:.6e}")
        
        return {
            'chi': self.chi,
            'loss_history': self.loss_history,
            'chi_history': [record['chi'] for record in self.chi_record],
            'converged': self.converged,
            'final_loss': loss_after_chi,
            'n_iterations': 1            
        }