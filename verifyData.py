import os
import numpy as np
import pandas as pd

# Adjust if your paths are different
base_output_dir = "Data\\Simulation_Data"
output_dir = os.path.join(base_output_dir, "Simulation_Output")

coords_path = os.path.join(output_dir, "fixed_node_coords.npy")
nodes_path = os.path.join(output_dir, "fixed_node_nums.npy")
conn_path = os.path.join(output_dir, "fixed_connectivity.npy")
disp_path = os.path.join(output_dir, "all_displacement_tensors.npy")
csv_path = os.path.join(output_dir, "simulation_input_metadata.csv")

print("=== Loading files ===")
node_coords = np.load(coords_path)
node_nums = np.load(nodes_path)
connectivity = np.load(conn_path)
all_displacements = np.load(disp_path)
meta = pd.read_csv(csv_path)

print("\n=== Shapes & basic info ===")
print(f"node_coords.shape        = {node_coords.shape}")       # (N_nodes, 3)
print(f"node_nums.shape          = {node_nums.shape}")         # (N_nodes,)
print(f"connectivity.shape       = {connectivity.shape}")      # (N_elems, nodes_per_elem)
print(f"all_displacements.shape  = {all_displacements.shape}") # (N_cases, N_nodes, 3)
print(f"meta.shape               = {meta.shape}")              # (N_cases, 5)

print("\n=== Metadata preview (first 5 rows) ===")
print(meta.head())

# Consistency checks
n_cases, n_nodes_disp, n_comp = all_displacements.shape
n_nodes_coords = node_coords.shape[0]
n_nodes_nums = node_nums.shape[0]

print("\n=== Consistency checks ===")
print(f"Nodes: disp={n_nodes_disp}, coords={n_nodes_coords}, nums={n_nodes_nums}")
if n_nodes_disp == n_nodes_coords == n_nodes_nums:
    print("✅ Node counts are consistent.")
else:
    print("⚠️ Node counts are NOT consistent. Check meshing vs. saved data.")

if n_cases == len(meta):
    print(f"✅ Number of cases matches metadata: {n_cases}")
else:
    print(f"⚠️ Mismatch: displacements have {n_cases} cases, "
          f"metadata has {len(meta)} rows.")
