#%%
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

N_Gaussians = 3

# Generate the Gaussian target data
x_data = np.linspace(-10, 10, 200)
true_params = [3, -5, 1, 5, 1, 0.5, 2, 6, 0.75]  # [A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3]
y_data = sum(gaussian(x_data, true_params[i], true_params[i+1], true_params[i+2]) for i in range(0, len(true_params), 3))

def mean_value_loss_function(mu_hats, mu_obs):
    mu_hat_1, mu_hat_2, mu_hat_3 = mu_hats
    mu_obs_1, mu_obs_2, mu_obs_3 = mu_obs
    
    # Assume generic standard Gaussians, since only optimizing for their locations here
    A = 1
    sigma = 1
    
    # Compute the predicted Gaussian functions
    y_hat_1 = gaussian(x_data, A, mu_hat_1, sigma)
    y_hat_2 = gaussian(x_data, A, mu_hat_2, sigma)
    y_hat_3 = gaussian(x_data, A, mu_hat_3, sigma)
    
    # Normalize the Gaussians
    normalized_y_hat_1 = y_hat_1 / np.sum(y_hat_1)
    normalized_y_hat_2 = y_hat_2 / np.sum(y_hat_2)
    normalized_y_hat_3 = y_hat_3 / np.sum(y_hat_3)
    
    # Compute their empirical means
    Ey_1 = np.dot(normalized_y_hat_1, x_data)
    Ey_2 = np.dot(normalized_y_hat_2, x_data)
    Ey_3 = np.dot(normalized_y_hat_3, x_data)
    
    # Compute the loss as the sum of squared differences between predicted and observed means
    loss = (Ey_1 - mu_obs_1) ** 2 + (Ey_2 - mu_obs_2) ** 2 + (Ey_3 - mu_obs_3) ** 2
    return loss

def morphology_loss_function(morph_ests, mu_hats):
    """
        Here we assume mu_hats are known from the previous mean value optimization step,
        and we want to optimize the morphology estimates (e.g., scales and widths) to fit the target data.
    """
    scale_1, width_1, scale_2, width_2, scale_3, width_3 = morph_ests # Should be a np.array (initial guess)
    mu_hat_1, mu_hat_2, mu_hat_3 = mu_hats # Should be a tuple
    
    # Compute the predicted Gaussian functions
    y_hat_1 = gaussian(x_data, scale_1, mu_hat_1, width_1)
    y_hat_2 = gaussian(x_data, scale_2, mu_hat_2, width_2)
    y_hat_3 = gaussian(x_data, scale_3, mu_hat_3, width_3)
    
    # Sum the predicted Gaussians to get the overall prediction
    y_hat = y_hat_1 + y_hat_2 + y_hat_3

    # Compute the loss as the mean squared error between predicted and target data
    loss = np.mean((y_hat - y_data) ** 2)
    return loss

# Location optimization step
mu_obs = (true_params[1], true_params[4], true_params[7])  # True means from the target data
rec_res = []; errors = []
dummy_morph = (1, 1, 1, 1, 1, 1)  # Dummy morphology values for mean optimization

def mean_callback(xk):
    # Record all 9 parameters: interleave means with dummy morphology
    full_params = np.array([dummy_morph[0], xk[0], dummy_morph[1], 
                            dummy_morph[2], xk[1], dummy_morph[3],
                            dummy_morph[4], xk[2], dummy_morph[5]])
    rec_res.append(full_params.copy())
    errors.append(mean_value_loss_function(xk, mu_obs))

# Add initial guess to rec_res
mean_callback(np.array([0, 0, 0]))

res = minimize(mean_value_loss_function, x0=(0, 0, 0), args=(mu_obs,), method='Nelder-Mead', callback=mean_callback)
n_mean_evals = len(rec_res)

# Morphology optimization step
def morph_callback(xk):
    # Record all 9 parameters: keep optimized means, update morphology
    full_params = np.array([xk[0], res.x[0], xk[1],
                            xk[2], res.x[1], xk[3],
                            xk[4], res.x[2], xk[5]])
    rec_res.append(full_params.copy())
    errors.append(morphology_loss_function(xk, res.x))

morph_res = minimize(morphology_loss_function, x0=(1, 1, 1, 1, 1, 1), args=(res.x,), method='Nelder-Mead', callback=morph_callback)
total_evals = len(rec_res)

# Convert rec_res to list of DataFrames
rec_res_df = []
for params in rec_res:
    df = pd.DataFrame({
        'A': [params[0], params[3], params[6]],
        'mu': [params[1], params[4], params[7]],
        'sigma': [params[2], params[5], params[8]],
        'Gaussian_index': [0, 1, 2]
    })
    rec_res_df.append(df)

# %%
# Export the optimization as an mp4 animation using Matplotlib.Animation
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(-10, 10)
ax.set_ylim(0, 4)
ax.set_xlabel('x')
ax.set_ylabel('y')
frame_counter = 0

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global frame_counter
    df = frame
    y_hat = sum(gaussian(x_data, df.loc[i, 'A'], df.loc[i, 'mu'], df.loc[i, 'sigma']) 
                for i in range(len(df)))
    line.set_data(x_data, y_hat)
    line.set_label('Fit')
    
    # Also plot the target data for reference
    y_true = sum(gaussian(x_data, true_params[i], true_params[i+1], true_params[i+2]) for i in range(0, len(true_params), 3))
    ax.plot(x_data, y_true, color='black', linestyle='--', label='Target Data' if frame is rec_res_df[0] else "")
    ax.legend()
        
    # Set the axis limits to the global maximum values
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 6)
    
    frame_counter += 1; print(frame_counter)
    if frame_counter < n_mean_evals:
        ax.set_title(f'Gaussian Fit Optimization: Mean Value Step {frame_counter}/{total_evals}; MSE={errors[frame_counter-1]:.4f}')
    else:
        ax.set_title(f'Gaussian Fit Optimization: Morphology Step {frame_counter}/{total_evals}; MSE={errors[frame_counter-1]:.4f}')
    return line,

ani = FuncAnimation(fig, update, frames=rec_res_df, init_func=init, blit=True, repeat=False)
ani.save('gaussian_fit_optimization.mp4', writer='ffmpeg', fps=10)


#%%
class SusceptibilityTensorFitter:
    """
        Class to fit the susceptibility tensor chi given pseudocontact shift data and hyperfine tensors.
        Here we are assuming the correct assignment of the hyperfine tensors to atoms is known.
    """
    def __init__(self, hyperfines_dict, delta_pc_dict, chi_hat_init=None, known_assignment=True):
        if known_assignment:
            self.hyperfines_dict = hyperfines_dict              # Dictionary of hyperfine tensors A
        else:
            # Shuffle the values to simulate unknown assignment
            copied_vals = list(hyperfines_dict.values()).copy()
            np.random.shuffle(copied_vals)
            self.hyperfines_dict = {key: val for key, val in zip(hyperfines_dict.keys(), copied_vals)}
            
        self.y = np.array(list(delta_pc_dict.values()))                                 # Observed pseudocontact shifts
        self.chi_hat = chi_hat_init if chi_hat_init is not None else np.zeros(3, 3)     # Initial guess for chi
    
    def forward_model(self, chi):
        """
            Given a susceptibility tensor chi and a dictionary of hyperfine tensors, compute the pseudocontact shifts.
        """
        delta_pc_values = []
        for _, A in self.hyperfines_dict.items():
            value = (1/3) * np.trace(A @ chi)
            delta_pc_values.append(value)
        return np.array(delta_pc_values)

    def loss(self, chi_hat):
        """
            Loss function to minimize: mean squared error between predicted and observed pseudocontact shifts.
        """
        chi_hat = np.diag(chi_hat)
        y_hat = self.forward_model(chi_hat)
        loss_val = np.mean((y_hat - self.y) ** 2)
        return loss_val
    
    def fit(self):
        """
            Fit the susceptibility tensor chi to minimize the loss function.
        """
        chi_hat_diag = np.diag(self.chi_hat)
        res = minimize(self.loss, chi_hat_diag, method='Nelder-Mead')
        return res
    
    
# %%
# Test the alternating optimisation approach
# print("\n" + "="*60)
# print("ALTERNATING OPTIMISATION APPROACH")
# print("="*60)

# alt_fitter = AlternatingOptimisationFitter(
#     hyperfines_dict, 
#     delta_pc_dict, 
#     max_iters=50, 
#     tol=1e-8
# )
# result = alt_fitter.fit()

# print("\n" + "="*60)
# print("RESULTS COMPARISON")
# print("="*60)

# print("\nFinal assignment found:")
# for atom, original in result['assignment_dict'].items():
#     print(f"  {atom} ← hyperfine originally from {original}")

# print(f"\nFinal MSE: {result['final_loss']:.6e}")
# print(f"Converged: {result['converged']}")

# Compare predictions
# print("\n" + "="*60)
# print("SHIFT PREDICTIONS")
# print("="*60)

# print(f"\n{'Atom':<10} {'Observed':<12} {'Alt-Opt Pred':<12} {'Error':<12}")
# print("-"*50)

# alt_fitted_deltas = []
# true_deltas = []

# for atom_label in delta_pc_dict.keys():
#     A = alt_fitter.hyperfines_dict[atom_label]  # Using final assignment
#     fitted_value = (1/3) * np.trace(A @ result['chi'])
#     true_value = delta_pc_dict[atom_label]
#     error = fitted_value - true_value
    
#     print(f"{atom_label:<10} {true_value:>10.4f}  {fitted_value:>10.4f}  {error:>10.4f}")
    
#     alt_fitted_deltas.append(fitted_value)
#     true_deltas.append(true_value)

# Plot convergence
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# # Loss convergence
# ax1.plot(result['loss_history'], 'o-', linewidth=2, markersize=6)
# ax1.set_xlabel('Iteration', fontsize=12)
# ax1.set_ylabel('MSE Loss', fontsize=12)
# ax1.set_title('Alternating Optimization Convergence', fontsize=14, fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.set_yscale('log')

# # Fitted vs observed shifts
# ax2.scatter(true_deltas, alt_fitted_deltas, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
# ax2.plot([min(true_deltas), max(true_deltas)], [min(true_deltas), max(true_deltas)], 'r--', linewidth=2, label='Perfect fit')
# ax2.set_xlabel('Observed δ_pc (ppm)', fontsize=12)
# ax2.set_ylabel('Predicted δ_pc (ppm)', fontsize=12)
# ax2.set_title('Alternating Opt: Predicted vs Observed', fontsize=14, fontweight='bold')
# ax2.legend(fontsize=10)
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()