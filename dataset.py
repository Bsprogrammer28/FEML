# ============================================================
# dataset.py
# FEM Graph Dataset with Normalization
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

# ----------------------------
# Normalization constants
# ----------------------------
BC_TOL = 1e-3

LENGTH = 10.0          # beam length (m)
MAX_FORCE = 10000.0    # max applied force (N)
E_REF = 200e9          # reference Young's modulus (Pa)


def build_case_features(
    coords,
    bc_fixed,
    node_nums,
    applied_node_id,
    force,
    load_x,
    E=200e9,
    nu=0.3
):
    """
    Build normalized node features for ONE load case
    """
    N = coords.shape[0]

    # ---- Load mask ----
    load_mask = np.zeros(N, dtype=np.float32)
    node_map = {int(n): i for i, n in enumerate(node_nums)}
    load_mask[node_map[int(applied_node_id)]] = 1.0

    # ---- Normalize coordinates ----
    coords_norm = coords.copy()
    coords_norm[:, 0] /= LENGTH
    coords_norm[:, 1] /= LENGTH
    coords_norm[:, 2] /= LENGTH

    return np.concatenate([
        coords_norm,                         # (N,3) normalized xyz
        bc_fixed[:, None],                   # (N,1) BC mask
        load_mask[:, None],                  # (N,1) load mask
        np.full((N,1), force / MAX_FORCE),   # normalized force
        np.full((N,1), load_x / LENGTH),     # normalized load position
        np.full((N,1), E / E_REF),            # normalized Young's modulus
        np.full((N,1), nu)                    # Poisson ratio
    ], axis=1)


class FEMGraphDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        # ---- Fixed mesh data ----
        self.coords = np.load(os.path.join(data_dir, "fixed_node_coords.npy"))
        self.node_nums = np.load(os.path.join(data_dir, "fixed_node_nums.npy"))

        self.edge_index = torch.tensor(
            np.load(os.path.join(data_dir, "GNN_Edge_Index_A.npy")),
            dtype=torch.long
        )

        # ---- Load metadata ----
        self.meta = pd.read_csv(
            os.path.join(data_dir, "simulation_input_metadata.csv")
        )

        # ---- Displacements (target) ----
        self.displacements = np.load(
            os.path.join(data_dir, "all_displacement_tensors.npy")
        )

        # ---- Normalize displacements ----
        self.disp_scale = np.max(np.abs(self.displacements))
        self.displacements = self.displacements / self.disp_scale

        # Save scale for inference / visualization
        np.save(
            os.path.join(data_dir, "disp_scale.npy"),
            self.disp_scale
        )

        # ---- Boundary condition mask ----
        self.bc_fixed = (np.abs(self.coords[:, 0]) < BC_TOL).astype(np.float32)

    def len(self):
        return len(self.meta)

    def get(self, idx):
        row = self.meta.iloc[idx]

        X = build_case_features(
            self.coords,
            self.bc_fixed,
            self.node_nums,
            row["Applied_Node_ID"],
            row["Load_Magnitude_N"],
            row["Load_Position_X_m"]
        )

        y = self.displacements[idx]

        return Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=self.edge_index,
            y=torch.tensor(y, dtype=torch.float32)
        )
