import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# --- 1. LOAD FIXED MESH DATA ---
# Using the filenames from your simulation script
node_coords = np.load("Data\\Simulation_Data\\Simulation_Output\\fixed_node_coords.npy")  # (N, 3) -> x, y, z
node_nums = np.load("Data\\Simulation_Data\\Simulation_Output\\fixed_node_nums.npy")      # (N,)   -> ANSYS node numbers
# connectivity = np.load("fixed_connectivity.npy") # Element-node mapping

# If you already have the edge_index (from a previous conversion step):
edge_index_t = torch.tensor(np.load("Data\\Simulation_Data\\Simulation_Output\\GNN_Edge_Index_A.npy"), dtype=torch.long)
# --- 2. LOAD SIMULATION RESULTS ---
df_metadata = pd.read_csv("Data\\Simulation_Data\\Simulation_Output\\simulation_input_metadata.csv")
try:
    Y_targets = np.load("Data\\Simulation_Data\\Simulation_Output\\all_displacement_tensors.npy")
except FileNotFoundError:
    print("Warning: Displacement targets not found. Using dummy targets.")
    Y_targets = np.zeros((len(df_metadata), node_coords.shape[0], 3))

# --- 3. PHYSICS-AWARE DATA CONVERSION ---

def create_physics_aware_dataset(coords, node_ids, metadata, targets, edge_index):
    data_list = []
    num_nodes = coords.shape[0]
    
    # Pre-calculate Boundary Condition (BC) Flag
    # Nodes at X=0 are fixed. We use a small tolerance.
    is_fixed = (coords[:, 0] < 0.001).astype(np.float32).reshape(-1, 1)
    
    # Create a mapping from ANSYS Node ID to array index
    # (Because Applied_Node_ID in CSV refers to the ANSYS number)
    id_to_idx = {int(node_id): i for i, node_id in enumerate(node_ids)}
    
    for i in range(len(metadata)):
        row = metadata.iloc[i]
        
        # A. Get Case-Specific Load Info
        applied_node_id = int(row['Applied_Node_ID'])
        load_magnitude = float(row['Load_Magnitude_N'])
        
        # B. Create the Sparse Load Feature
        # Every node has 0 load feature except the one being pulled
        load_feature = np.zeros((num_nodes, 1), dtype=np.float32)
        if applied_node_id in id_to_idx:
            target_idx = id_to_idx[applied_node_id]
            load_feature[target_idx] = load_magnitude
        
        # C. Concatenate Features: [x, y, z, BC_Fixed, Load_Val]
        # Total input channels = 5
        x_coords_t = torch.tensor(coords, dtype=torch.float)
        bc_t = torch.tensor(is_fixed, dtype=torch.float)
        load_t = torch.tensor(load_feature, dtype=torch.float)
        
        x_i = torch.cat([x_coords_t, bc_t, load_t], dim=1)
        
        # D. Target Displacement (dx, dy, dz)
        y_i = torch.tensor(targets[i], dtype=torch.float)
        
        # E. Assemble PyG Data Object
        data = Data(x=x_i, edge_index=edge_index, y=y_i)
        data_list.append(data)
        
        if i % 1000 == 0:
            print(f"Processed {i} / {len(metadata)} cases...")

    return data_list

# Generate the dataset
gnn_dataset = create_physics_aware_dataset(
    node_coords, node_nums, df_metadata, Y_targets, edge_index_t
)

print(f"\nCreated {len(gnn_dataset)} data objects.")
print(f"Node feature shape: {gnn_dataset[0].x.shape}") # Should be (N, 5)