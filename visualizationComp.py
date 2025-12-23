# ============================================================
# visualize.py
# Compare ANSYS vs GNN deformation
# ============================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from dataset import FEMGraphDataset
from model import FEMGNN

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "Data/Simulation_Data/Simulation_Output"
MODEL_PATH = "fem_gnn_model.pt"
CASE_INDEX = 0           # which load case to visualize
DEFORM_SCALE = 50        # visual exaggeration factor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD DATA & MODEL
# ----------------------------
dataset = FEMGraphDataset(DATA_DIR)
data = dataset[CASE_INDEX].to(DEVICE)

model = FEMGNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------------
# PREDICTION
# ----------------------------
with torch.no_grad():
    u_pred_norm = model(data)

# ---- De-normalize displacement ----
disp_scale = np.load(os.path.join(DATA_DIR, "disp_scale.npy"))

u_pred = u_pred_norm.cpu().numpy() * disp_scale
u_true = data.y.cpu().numpy() * disp_scale

coords = dataset.coords

# ----------------------------
# COMPUTE ERROR
# ----------------------------
err = np.linalg.norm(u_pred - u_true, axis=1)

print("Max ANSYS displacement:", np.linalg.norm(u_true, axis=1).max())
print("Max GNN displacement  :", np.linalg.norm(u_pred, axis=1).max())
print("Max nodal error       :", err.max())

# ----------------------------
# DEFORMED COORDINATES
# ----------------------------
coords_ansys = coords + DEFORM_SCALE * u_true
coords_gnn = coords + DEFORM_SCALE * u_pred

# ----------------------------
# PLOTTING
# ----------------------------
fig = plt.figure(figsize=(18, 5))

# ---- Original mesh ----
ax1 = fig.add_subplot(131, projection="3d")
ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2)
ax1.set_title("Original Mesh")

# ---- ANSYS deformation ----
ax2 = fig.add_subplot(132, projection="3d")
p2 = ax2.scatter(
    coords_ansys[:, 0],
    coords_ansys[:, 1],
    coords_ansys[:, 2],
    c=np.linalg.norm(u_true, axis=1),
    cmap="viridis",
    s=2
)
ax2.set_title("ANSYS Deformation")
fig.colorbar(p2, ax=ax2, shrink=0.6, label="|u| (m)")

# ---- GNN deformation ----
ax3 = fig.add_subplot(133, projection="3d")
p3 = ax3.scatter(
    coords_gnn[:, 0],
    coords_gnn[:, 1],
    coords_gnn[:, 2],
    c=np.linalg.norm(u_pred, axis=1),
    cmap="viridis",
    s=2
)
ax3.set_title("GNN Predicted Deformation")
fig.colorbar(p3, ax=ax3, shrink=0.6, label="|u| (m)")

plt.tight_layout()
plt.show()

# ----------------------------
# ERROR PLOT
# ----------------------------
plt.figure(figsize=(6, 4))
plt.scatter(
    coords[:, 0],
    err,
    s=5
)
plt.xlabel("X coordinate (m)")
plt.ylabel("Nodal Error (m)")
plt.title("Displacement Error Distribution")
plt.grid(True)
plt.show()
