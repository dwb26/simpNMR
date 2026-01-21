#%%
import pandas as pd
import numpy as np

def load_hyperfines_and_susceptibility_tensor(file_path="", verbose=False):

    # Read in the hyperfines and shifts file
    hyperfines_and_shifts_df = pd.read_csv('hyperfines_and_shifts_298.00_K.csv')

    # Read in the susceptibility tensor data
    susceptibility_tensor_data = pd.read_csv('susceptibility_tensor.csv').columns.tolist()
    susceptibility_tensor_columns = "Temperature (K), chi_iso (Å^3), chi_xx (Å^3), chi_xy (Å^3), chi_xz (Å^3), chi_yy (Å^3), chi_yz (Å^3), chi_zz (Å^3), dchi_xx (Å^3), dchi_xy (Å^3), dchi_xz (Å^3), dchi_yy (Å^3), dchi_yz (Å^3), dchi_zz (Å^3), chi_x (Å^3), chi_y (Å^3), chi_z (Å^3), chi_ax (Å^3), chi_rho (Å^3), alpha (degrees), beta (degrees), gamma (degrees)"
    susceptibility_tensor_columns = list(susceptibility_tensor_columns.split(", "))
    data_dict = {key: val for key, val in zip(susceptibility_tensor_columns, susceptibility_tensor_data)}
    susceptibility_tensor_df = pd.DataFrame(data_dict, index=[0])

    # First construct the susceptibility tensor chi as a 3x3 matrix
    chi = np.array([[susceptibility_tensor_df.at[0, 'chi_xx (Å^3)'], susceptibility_tensor_df.at[0, 'chi_xy (Å^3)'], susceptibility_tensor_df.at[0, 'chi_xz (Å^3)']],
                    [susceptibility_tensor_df.at[0, 'chi_xy (Å^3)'], susceptibility_tensor_df.at[0, 'chi_yy (Å^3)'], susceptibility_tensor_df.at[0, 'chi_yz (Å^3)']],
                    [susceptibility_tensor_df.at[0, 'chi_xz (Å^3)'], susceptibility_tensor_df.at[0, 'chi_yz (Å^3)'], susceptibility_tensor_df.at[0, 'chi_zz (Å^3)']]])
    chi = np.array([[-0.02903, 0.00000, 0.00000], 
                    [0.00000, -0.0774, 0.00000], 
                    [0.00000, 0.00000, 0.10643]])
    
    # Axial and rhombic components
    chi_ax = 0.15965
    chi_rho = 0.02419
    delta_chi_x, delta_chi_y, delta_chi_z = np.diag(chi)
    
    # Assert that chi is traceless
    trace_chi = np.trace(chi)
    assert np.isclose(trace_chi, 0.0), f"Susceptibility tensor chi is not traceless! Trace = {trace_chi}"
    
    print(chi_ax - 1.5 * delta_chi_z)
    print(chi_rho - 0.5 * (delta_chi_x - delta_chi_y))

    # Next, for each atom in the hyperfines_and_shifts_df, store their hyperfine tensor A as a 3x3 matrix
    hyperfine_tensors_dict = {}
    for _, row in hyperfines_and_shifts_df.iterrows():
        atom_label = row['atom_label ()']
        A = [row['Adip_xx (ppm Å^-3)'], row['Adip_xy (ppm Å^-3)'], row['Adip_xz (ppm Å^-3)'], row['Adip_yy (ppm Å^-3)'], row['Adip_yz (ppm Å^-3)'], row['Adip_zz (ppm Å^-3)']]
        
        # Convert A to a symmetric matrix
        A = np.array([[A[0], A[1], A[2]],
                    [A[1], A[3], A[4]],
                    [A[2], A[4], A[5]]])
        hyperfine_tensors_dict[atom_label] = A

    # For each hyperfine tensor A, compute 1/3 * trace(A * chi)
    delta_pc_dict = {}
    for atom_label, A in hyperfine_tensors_dict.items():
        value = (1/3) * np.trace(A @ chi)
        delta_pc_dict[atom_label] = value
    
    if verbose:
        print("Hyperfines and shifts df")
        print("------------------------")
        print(hyperfines_and_shifts_df.head())

        print("\nSusceptibility tensor df")
        print("------------------------")
        print(susceptibility_tensor_df.head())
        
        print("Susceptibility tensor chi:")
        print(chi)
        
        print("\nHyperfine tensors A for each atom:")
        for atom_label, A in hyperfine_tensors_dict.items():
            print(f"\nAtom: {atom_label}")
            print(A)
    
    return hyperfine_tensors_dict, chi

def load_observed_pseudocontact_shift_data():
    # Read in the hyperfines and shifts file
    hyperfines_and_shifts_df = pd.read_csv('hyperfines_and_shifts_298.00_K.csv')
    
    delta_pc_dict = {}
    for _, row in hyperfines_and_shifts_df.iterrows():
        atom_label = row['atom_label ()']
        delta_pc = row['δ_pc (ppm)']
        delta_pc_dict[atom_label] = delta_pc
    return delta_pc_dict
 
# res = load_hyperfines_and_susceptibility_tensor()