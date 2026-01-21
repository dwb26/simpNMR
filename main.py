#%%
from data_handling import load_hyperfines_and_susceptibility_tensor, load_observed_pseudocontact_shift_data
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

hyperfines_dict, chi_true = load_hyperfines_and_susceptibility_tensor()
delta_pc_dict = load_observed_pseudocontact_shift_data()

class AlternatingOptimisationFitter:
    """
    Alternating optimization approach:
    - Step 1: Fix assignment, optimise χ
    - Step 2: Fix χ, optimise assignment (using Hungarian algorithm)
    - Repeat until convergence
    
    This avoids exhaustive enumeration of all N! permutations.
    """
    def __init__(self, hyperfines_dict, delta_pc_dict, chi_iso=0.18013, chi_init=None, max_iters=50, tol=1e-6):
        """
        Parameters:
        -----------
        hyperfines_dict : dict
            Dictionary mapping atom labels to hyperfine tensors (3x3 arrays)
        delta_pc_dict : dict
            Dictionary mapping atom labels to observed pseudocontact shifts
        chi_init : np.ndarray, optional
            Initial guess for susceptibility tensor (3x3). If None, uses zeros.
        max_iters : int
            Maximum number of alternating iterations
        tol : float
            Convergence tolerance for loss change
        """
        self.atom_labels = list(delta_pc_dict.keys())
        self.N_atoms = len(self.atom_labels)
        self.observed_shifts = np.array([delta_pc_dict[label] for label in self.atom_labels])
        
        # Store original hyperfines as list (we'll permute these)
        self.hyperfine_tensors_original = [hyperfines_dict[label] for label in self.atom_labels]
        
        # Initialise with random permutation to model unknown assignment scenario
        self.current_permutation = np.random.permutation(len(self.atom_labels))
        self.hyperfines_dict = self._build_assignment(self.current_permutation)
        
        self.chi_iso = chi_iso if chi_iso is not None else 0.0
        self.chi = chi_init if chi_init is not None else np.zeros((3, 3))
        self.max_iters = max_iters
        self.tol = tol
        
        # Track convergence
        self.loss_history = []
        self.assignment_history = []
        self.chi_history = []
        self.frame_data = []  # Store data for animation        
    
    def forward_model(self, chi):
        """
            Given a susceptibility tensor chi and a dictionary of hyperfine tensors, compute the pseudocontact shifts.
        """
        y_hat = np.array([
            (1/3) * np.trace(self.hyperfines_dict[label] @ chi)
            for label in self.atom_labels
        ])
        return y_hat    
    
    def loss_fn(self, chi_diag):
        """
            Loss function to minimize: mean squared error between predicted and observed pseudocontact shifts.
        """
        chi_hat = np.diag(chi_diag)
        y_hat = self.forward_model(chi_hat)
        loss_val = np.mean((y_hat - self.observed_shifts)**2)
        return loss_val
    
    def z_constraint(self, chi_diag):
        chi_z = chi_diag[2]
        
        # Constrain that -chi_iso < chi_z and chi_z < 2*chi_iso
        return [chi_z + self.chi_iso, 2*self.chi_iso - chi_z]
    
    def quotient_constraint(self, chi_diag):
        chi_x = chi_diag[0]
        chi_y = chi_diag[1]
        chi_z = chi_diag[2]
        
        # Constrain that 0 < (chi_x - chi_y)/chi_z < 1
        ratio = (chi_x - chi_y) / chi_z if chi_z != 0 else 0.0
        return [ratio, 1 - ratio]
    
    def trace_constraint(self, chi_diag):
        chi_x = chi_diag[0]
        chi_y = chi_diag[1]
        chi_z = chi_diag[2]
        
        # Constrain that trace(chi) = 0
        # return [-(chi_x + chi_y + chi_z), chi_x + chi_y + chi_z]
        return [-(chi_x + chi_y + chi_z)]
    
    def fit_chi(self):
        """
            Fix assignment, optimise χ
        """
        intermediate_results = []
        def callback_fn(xk, y=None):
            intermediate_results.append(xk)
        
        constraints = [
            {'type': 'ineq', 'fun': self.z_constraint},
            {'type': 'ineq', 'fun': self.quotient_constraint},
            {'type': 'eq', 'fun': self.trace_constraint}
        ]
        # res = minimize(self.loss_fn, np.diag(self.chi), method='SLSQP', constraints=constraints, callback=callback_fn)
        res = minimize(self.loss_fn, 
                       np.diag(self.chi), 
                       method='trust-constr', 
                       constraints=constraints, 
                       callback=callback_fn)
        
        self.chi = np.diag(res.x)

    def optimise_assignment(self):
        """
            Fix χ, optimise assignment using Hungarian algorithm
        """
        cost_matrix = np.zeros((self.N_atoms, self.N_atoms))
        
        # Build cost matrix: C[i,j] = cost of assigning hyperfine j to atom i
        for i in range(self.N_atoms):
            for j in range(self.N_atoms):
                predicted_shift = (1/3) * np.trace(self.hyperfine_tensors_original[j] @ self.chi)
                cost_matrix[i, j] = (self.observed_shifts[i] - predicted_shift)**2
        
        # Solve linear assignment problem
        _, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update permutation and assignment
        self.current_permutation = col_ind
        self.hyperfines_dict = self._build_assignment(col_ind)
        
        return col_ind    
    
    def fit(self):
        """
        Run alternating optimisation until convergence.
        
        Returns:
        --------
        dict with keys:
            'chi': fitted susceptibility tensor
            'assignment': final permutation
            'loss_history': loss at each iteration
            'converged': whether algorithm converged
        """
        
        # Store initial state (frame 0: random assignment, zero chi)
        predicted_shifts_init = self.forward_model(self.chi)
        self.frame_data.append({
            'iteration': 0,
            'step': 'initial',
            'chi': self.chi.copy(),
            'assignment': self.current_permutation.copy(),
            'predicted': predicted_shifts_init.copy(),
            'observed': self.observed_shifts.copy(),
            'loss': self.loss_fn(np.diag(self.chi))
        })
        
        for iteration in range(self.max_iters):
            
            # Step 1: Optimise χ given assignment
            self.fit_chi()
            loss_after_chi = self.loss_fn(np.diag(self.chi))
            
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
            
            # Step 2: Optimise assignment given χ
            new_assignment = self.optimise_assignment()
            loss_after_assignment = self.loss_fn(np.diag(self.chi))
            
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
            
            # Check convergence
            if iteration > 0:
                loss_change = abs(self.loss_history[-2] - self.loss_history[-1])
                if loss_change < self.tol:
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
            'converged': iteration < self.max_iters - 1,
            'final_loss': self.loss_history[-1]
        }
        
    def _build_assignment(self, permutation):
        """
            Build hyperfines_dict from permutation indices
        """
        return {
            self.atom_labels[i]: self.hyperfine_tensors_original[permutation[i]]
            for i in range(len(self.atom_labels))
        }

print("\n" + "="*60)
print("MULTIPLE RANDOM INITIALIZATIONS")
print("="*60)

n_trials = 10
results_list = []

print(f"\nRunning {n_trials} trials with different random seeds...\n")

chi_iso = 0.18013
chi_z = 1.1 * chi_iso
chi_x = chi_iso + (chi_iso / 2)
chi_y = chi_iso - (chi_iso / 2)
chi_init = np.diag([chi_x, chi_y, chi_z])

for trial in range(n_trials):
    np.random.seed(trial)  # For reproducibility
    
    alt_fitter = AlternatingOptimisationFitter(
        hyperfines_dict, 
        delta_pc_dict, 
        chi_init=chi_init,
        max_iters=50,
        tol=1e-8
    )
    
    result = alt_fitter.fit()
    results_list.append(result)
    
    # Test if the output result satisfies constraints
    chi_fitted_diag = np.diag(result['chi'])
    z_cons = alt_fitter.z_constraint(chi_fitted_diag)
    quot_cons = alt_fitter.quotient_constraint(chi_fitted_diag)
    trace_cons = alt_fitter.trace_constraint(chi_fitted_diag)
    assert all(c >= 0 for c in z_cons), f"Trial {trial+1}: z_constraint violated in final result: {z_cons}"
    assert all(c >= 0 for c in quot_cons), f"Trial {trial+1}: quotient_constraint violated in final result: {quot_cons}"
    assert all(np.isclose(c, 0) for c in trace_cons), f"Trial {trial+1}: trace_constraint violated in final result: {trace_cons}"
    
    print(f"Trial {trial+1:2d}: Final MSE = {result['final_loss']:.6e}, "
          f"Converged = {result['converged']}, "
          f"Iterations = {len(result['loss_history'])}")
    
    # Compare the true chi with fitted chi
    chi_fitted = result['chi']
    chi_error = np.linalg.norm(chi_fitted - chi_true)
    print(f"          \nTrue χ:\n{chi_true}")
    print(f"          \nFitted χ:\n{chi_fitted}")
    print(f"||χ_fitted - χ_true||_F = {chi_error:.6e}\n")

# Find best result
best_idx = np.argmin([r['final_loss'] for r in results_list])
best_result = results_list[best_idx]


#%%
# ----------------------------------------------------------------------
#
# POST PROCESSING AND VISUALISATION OF RESULTS
#
# ----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"BEST RESULT: Trial {best_idx + 1}")
print(f"{'='*60}")
print(f"Final MSE: {best_result['final_loss']:.6e}")
print(f"\nBest fitted chi:")
print(best_result['chi'])
print(f"\nTrue chi:")
print(chi_true)

# Compare best result predictions
print("\n" + "="*60)
print("BEST RESULT: SHIFT PREDICTIONS")
print("="*60)

# Rebuild hyperfines_dict with best assignment
best_fitter = AlternatingOptimisationFitter(
    hyperfines_dict, 
    delta_pc_dict
)
best_fitter.hyperfines_dict = best_fitter._build_assignment(best_result['assignment'])
best_fitter.chi = best_result['chi']

# Measure the quality of the assignment by comparing the hyperfine tensors
for atom_label in hyperfines_dict.keys():
    A_hat = best_fitter.hyperfines_dict[atom_label]
    A_true = hyperfines_dict[atom_label]
    print("Atom: {}, Fitted A:\n{},\nTrue A:\n{}\n".format(atom_label, A_hat, A_true))

print(f"\n{'Atom':<10} {'Observed':<12} {'Predicted':<12} {'Error':<12}")
print("-"*50)

best_fitted_deltas = []
true_deltas = []
errors = []

for atom_label in delta_pc_dict.keys():
    A = best_fitter.hyperfines_dict[atom_label]
    fitted_value = (1/3) * np.trace(A @ best_result['chi'])
    true_value = delta_pc_dict[atom_label]
    error = fitted_value - true_value
    
    print(f"{atom_label:<10} {true_value:>10.4f}  {fitted_value:>10.4f}  {error:>10.4f}")
    
    best_fitted_deltas.append(fitted_value)
    true_deltas.append(true_value)
    errors.append(error)

rmse = np.sqrt(np.mean(np.array(errors)**2))

# Final comparison plot
plt.figure(figsize=(10, 8))
plt.scatter(true_deltas, best_fitted_deltas, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.plot([min(true_deltas), max(true_deltas)], [min(true_deltas), max(true_deltas)], 'r--', linewidth=2, label='Perfect fit')
plt.xlabel('Observed δ_pc (ppm)', fontsize=14)
plt.ylabel('Predicted δ_pc (ppm)', fontsize=14)
plt.title(f'Best Result: Predicted vs Observed\n RMSE = {rmse:.5f} ppm', fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, fancybox=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# ----------------------------------------------------------------------
#
# ANIMATION: Visualize the alternating optimization process
#
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("CREATING ANIMATION")
print("="*60)

# Use the best trial for animation
np.random.seed(best_idx)
animator_fitter = AlternatingOptimisationFitter(
    hyperfines_dict, 
    delta_pc_dict, 
    max_iters=50,
    tol=1e-8
)
animator_result = animator_fitter.fit()

print(f"\nAnimation will have {len(animator_fitter.frame_data)} frames")
print("Creating animation...")

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter([], [], s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

# Get global axis limits
all_observed = animator_fitter.observed_shifts
all_predicted_max = max([max(frame['predicted']) for frame in animator_fitter.frame_data])
all_predicted_min = min([min(frame['predicted']) for frame in animator_fitter.frame_data])

axis_min = min(all_observed.min(), all_predicted_min) * 1.1
axis_max = max(all_observed.max(), all_predicted_max) * 1.1

ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)
ax.set_xlabel('Observed δ_pc (ppm)', fontsize=14)
ax.set_ylabel('Predicted δ_pc (ppm)', fontsize=14)
ax.grid(True, alpha=0.3)

# Plot perfect fit line (constant)
ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=2, label='Perfect fit', zorder=1)

def init():
    scatter.set_offsets(np.empty((0, 2)))
    return scatter,

def update(frame_idx):
    frame = animator_fitter.frame_data[frame_idx]
    
    # Update scatter plot
    points = np.column_stack([frame['observed'], frame['predicted']])
    scatter.set_offsets(points)
    
    # Update title with iteration info
    if frame['step'] == 'initial':
        title = f"Initial: Random Assignment, χ = 0\nMSE = {frame['loss']:.4e}"
    elif frame['step'] == 'chi_optimized':
        title = f"Iteration {frame['iteration']}: After χ Optimization\nMSE = {frame['loss']:.4e}"
    else:  # assignment_optimized
        title = f"Iteration {frame['iteration']}: After Assignment Optimization (Hungarian)\nMSE = {frame['loss']:.4e}"
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Update legend
    ax.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, fancybox=True)
    
    return scatter,

# Create animation
ani = FuncAnimation(
    fig, 
    update, 
    frames=len(animator_fitter.frame_data),
    init_func=init, 
    blit=False, 
    repeat=True,
    interval=500  # 500ms between frames
)

# Save animation
output_file = 'alternating_optimization.mp4'
print(f"Saving animation to {output_file}...")
ani.save(output_file, writer='ffmpeg', fps=2, dpi=100)
print(f"Animation saved successfully!")

plt.close()

# %%
