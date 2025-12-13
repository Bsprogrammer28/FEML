import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv 
from torch_geometric.loader import DataLoader # For batch training

# --- 1. CONFIGURATION & DATA LOADING ---

# --- FIXED GRAPH COMPONENTS ---
# Ensure GNN files exist. If missing, try to generate them from the
# FEA export (scripts/generate_gnn_files.py). The generator writes to
# `Data\Simulation_Data\Simulation_Output` by default.
output_dir = os.path.join("Data", "Simulation_Data", "Simulation_Output")

def _ensure_gnn_files_and_get_paths():
    local_x = "GNN_Node_Features_X.npy"
    local_e = "GNN_Edge_Index_A.npy"

    # If both exist in CWD or in the expected output_dir, pick the available ones.
    cand_pairs = [
        (os.path.abspath(local_x), os.path.abspath(local_e)),
        (os.path.join(output_dir, local_x), os.path.join(output_dir, local_e)),
    ]

    for x_p, e_p in cand_pairs:
        if os.path.exists(x_p) and os.path.exists(e_p):
            return x_p, e_p

    # Not found — run generator script to create them.
    gen_script = os.path.join("scripts", "generate_gnn_files.py")
    if not os.path.exists(gen_script):
        raise FileNotFoundError(f"Generator script not found: {gen_script}")

    print("GNN files not found. Running generator: scripts/generate_gnn_files.py")
    subprocess.run([sys.executable, gen_script], check=True)

    # After generation, prefer files in output_dir
    x_p = os.path.join(output_dir, local_x)
    e_p = os.path.join(output_dir, local_e)
    if os.path.exists(x_p) and os.path.exists(e_p):
        return x_p, e_p

    # As fallback, check CWD
    if os.path.exists(local_x) and os.path.exists(local_e):
        return os.path.abspath(local_x), os.path.abspath(local_e)

    raise FileNotFoundError("GNN files were not produced by generator.")


X_path, edge_index_path = _ensure_gnn_files_and_get_paths()
X_coords = np.load(X_path) # (621, 3) -> (x, y, z)
edge_index_A = np.load(edge_index_path) # (2, 13797) -> Edges
# number of nodes
num_nodes = X_coords.shape[0]

# Convert fixed components to Tensors (outside the loop for efficiency)
x_fixed_coords = torch.tensor(X_coords, dtype=torch.float)
edge_index_t = torch.tensor(edge_index_A, dtype=torch.long)

# --- VARIABLE INPUTS (X) AND TARGETS (Y) ---
df_metadata = pd.read_csv("simulation_input_metadata.csv")
# Select the variable features: Load Position (Lx) and Load Magnitude (Fy)
X_global_var = df_metadata[['Load_Position_X_m', 'Load_Magnitude_N']].values # (10000, 2)
num_cases = X_global_var.shape[0]

# --- LOAD TARGETS (Y) ---
try:
    # Load the actual displacement results
    Y_targets = np.load("all_displacement_tensors.npy") 
    print(f"Successfully loaded Y targets (Displacements) with shape: {Y_targets.shape}")
except FileNotFoundError:
    # Fallback/Placeholder if the results file isn't found
    print("WARNING: 'all_displacement_tensors.npy' not found. Using MOCK data for demonstration.")
    Y_targets = np.random.rand(num_cases, num_nodes, 3).astype(np.float32) 
    

# --- 2. PHASE 2: DATA ASSEMBLY (FEA to PyG Data Objects) ---

def create_pyg_data_list(X_coords_t, edge_index_t, X_global_var, Y_targets, num_nodes):
    """
    Combines fixed coordinates, variable global inputs, and targets into 
    a list of PyTorch Geometric Data objects (one per simulation case).
    """
    data_list = []
    
    for i in range(Y_targets.shape[0]):
        
        # A. Get variable load inputs for case 'i'
        var_inputs_i = X_global_var[i, :].astype(np.float32) 
        
        # B. Broadcast Global Inputs to all nodes
        # Create a matrix of shape (621, 2) where every row is [Lx, Fy]
        x_global_i = torch.tensor(np.tile(var_inputs_i, (num_nodes, 1)), dtype=torch.float)
        
        # C. Concatenate to form the full input feature matrix X_i
        # Result: (621, 5) -> [x, y, z, Lx, Fy]
        x_i = torch.cat([X_coords_t, x_global_i], dim=1)
        
        # D. Target Y (Displacement dx, dy, dz)
        y_i = torch.tensor(Y_targets[i], dtype=torch.float)
        
        # E. Create the PyG Data Object
        data_i = Data(x=x_i, edge_index=edge_index_t, y=y_i)
        data_list.append(data_i)
        
    return data_list

# Generate the full dataset
gnn_dataset = create_pyg_data_list(x_fixed_coords, edge_index_t, X_global_var, Y_targets, num_nodes)

print(f"\n--- Phase 2 Complete ---")
print(f"Created {len(gnn_dataset)} PyTorch Geometric Data objects.")
print(f"Example Data Object (Case 1): {gnn_dataset[0]}")
print(f"Input Feature Shape (X): {gnn_dataset[0].x.shape} (Node Features)")
print(f"Target Label Shape (Y): {gnn_dataset[0].y.shape} (Nodal Displacements)")


# --- 3. PHASE 3: MODEL ARCHITECTURE (Encoder-Processor-Decoder GNN) ---

class EncoderProcessorDecoderGNN(torch.nn.Module):
    """
    GNN architecture for surrogate modeling of FEA results.
    The model predicts the 3D displacement vector (dx, dy, dz) for every node.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        super().__init__()
        
        # 1. Encoder (MLP) - Maps 5 input features to latent space
        self.encoder = Linear(in_channels, hidden_channels)
        
        # 2. Processor (GCN Layers) - Message Passing
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # GCNConv aggregates neighbor information
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 3. Decoder (MLP) - Maps latent features back to 3 DOFs
        self.decoder = Linear(hidden_channels, out_channels)
        
    def forward(self, data: Data):
        # Data unpacking
        x, edge_index = data.x, data.edge_index
        
        # 1. ENCODER
        x = self.encoder(x)
        x = F.relu(x)
        
        # 2. PROCESSOR
        # Information propagates across the mesh structure
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        # 3. DECODER
        # Final linear mapping to the 3 output dimensions (dx, dy, dz)
        x = self.decoder(x)
        
        return x

# --- MODEL INSTANTIATION ---
INPUT_FEATURES = 5  # (x, y, z, Load_Position_X_m, Load_Magnitude_N)
OUTPUT_DOFS = 3     # (dx, dy, dz)
HIDDEN_DIM = 64     # Latent space dimension
GNN_LAYERS = 4      # Number of message passing steps

model = EncoderProcessorDecoderGNN(
    in_channels=INPUT_FEATURES, 
    hidden_channels=HIDDEN_DIM, 
    num_layers=GNN_LAYERS, 
    out_channels=OUTPUT_DOFS
)

print(f"\n--- Phase 3 Complete ---")
print(f"GNN Model Defined:")
print(model)


# --- 4. PHASE 4: TRAINING SETUP (Conceptual) ---

# Splitting data (80% train, 20% test)
train_split = int(0.8 * num_cases)
train_dataset = gnn_dataset[:train_split]
test_dataset = gnn_dataset[train_split:]

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Use Mean Squared Error (MSE) as a loss for regression problems
criterion = torch.nn.MSELoss() 

print(f"\nPhase 4 Setup: Data Split and Loaders created.")
print(f"Training Cases: {len(train_dataset)}, Test Cases: {len(test_dataset)}")

# --- NEXT STEPS: TRAINING LOOP ---

# def train(model, loader, optimizer, criterion):
#     model.train()
#     total_loss = 0
#     for data in loader:
#         optimizer.zero_grad()
#         out = model(data)
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#     return total_loss / len(loader.dataset)

# # Example Training loop start:
# # for epoch in range(1, 101):
# #     loss = train(model, train_loader, optimizer, criterion)
# #     print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}')

# End of Code