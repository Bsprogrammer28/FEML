# ============================================================
# 3D Cantilever Beam – PINN vs ANSYS (PyAnsys)
# ============================================================

import numpy as np
import torch
import pyvista as pv

from ansys.mapdl.core import launch_mapdl
import torchphysics as tp

# ------------------------------------------------------------
# Geometry & material
# ------------------------------------------------------------
L, W, H = 1.0, 0.2, 0.2
E = 210e9
nu = 0.3
force_z = -1e5  # N (downward end load)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PART 1: ANSYS SOLUTION (REFERENCE)
# ============================================================

mapdl = launch_mapdl(run_location="ansys_temp", override=True)
mapdl.clear()
mapdl.prep7()

# Material
mapdl.mp("EX", 1, E)
mapdl.mp("PRXY", 1, nu)

# Element
mapdl.et(1, 186)

# Geometry
mapdl.block(0, L, 0, W, 0, H)

# Mesh
mapdl.esize(0.05)
mapdl.vmesh("ALL")

# Boundary conditions
mapdl.nsel("S", "LOC", "X", 0)
mapdl.d("ALL", "ALL", 0)

# End load
mapdl.nsel("S", "LOC", "X", L)
mapdl.f("ALL", "FZ", force_z / mapdl.get("_n", "NODE", 0, "COUNT"))

mapdl.allsel()

# Solve
mapdl.finish()
mapdl.run("/SOLU")
mapdl.solve()
mapdl.finish()

# Post-processing
mapdl.post1()
nodes = mapdl.mesh.nodes

ux = mapdl.post_processing.nodal_displacement("X")
uy = mapdl.post_processing.nodal_displacement("Y")
uz = mapdl.post_processing.nodal_displacement("Z")

disp_ansys = np.column_stack([ux, uy, uz])


# ============================================================
# PART 2: PINN INFERENCE
# ============================================================

# Rebuild PINN
X = tp.spaces.R1("x")
Y = tp.spaces.R1("y")
Z = tp.spaces.R1("z")
U = tp.spaces.R1("u")
V = tp.spaces.R1("v")
W_ = tp.spaces.R1("w")

model = tp.models.FCN(
    input_space=X * Y * Z,
    output_space=U * V * W_,
    hidden=(128, 128, 128),
    activations=torch.nn.Tanh()
).to(device)

model.load_state_dict(torch.load("beam_model3d.pth", map_location=device))
model.eval()

raw_net = next(model.children()).eval()

coords_t = torch.tensor(nodes, dtype=torch.float32, device=device)

with torch.no_grad():
    uvw = raw_net(coords_t)

disp_pinn = uvw.cpu().numpy()
disp_pinn_scaled = disp_pinn * 1e-3


# ============================================================
# PART 3: PYVISTA SIDE-BY-SIDE VISUALIZATION
# ============================================================

grid = mapdl.mesh.grid

grid["ANSYS_vec"] = disp_ansys
grid["PINN_vec"]  = disp_pinn_scaled

grid["ANSYS_disp"] = np.linalg.norm(disp_ansys, axis=1)
grid["PINN_disp"]  = np.linalg.norm(disp_pinn_scaled, axis=1)
grid["ERROR"]      = np.linalg.norm(disp_ansys - disp_pinn_scaled, axis=1)

warp_ansys = grid.warp_by_vector("ANSYS_vec", factor=50)
warp_pinn  = grid.warp_by_vector("PINN_vec", factor=50)

plotter = pv.Plotter(shape=(1, 3))

plotter.subplot(0, 0)
plotter.add_text("ANSYS", font_size=12)
plotter.add_mesh(
    warp_ansys,
    scalars="ANSYS_disp",
    cmap="viridis",
    show_edges=False
)

plotter.subplot(0, 1)
plotter.add_text("PINN", font_size=12)
plotter.add_mesh(
    warp_pinn,
    scalars="PINN_disp",
    cmap="viridis",
    show_edges=False
)

plotter.subplot(0, 2)
plotter.add_text("ABS ERROR", font_size=12)
plotter.add_mesh(
    grid,
    scalars="ERROR",
    cmap="inferno",
    show_edges=False
)

plotter.link_views()
plotter.show()  

# ============================================================
# PART 4: NUMERIC COMPARISON
# ============================================================

l2_error = np.linalg.norm(disp_ansys - disp_pinn) / np.linalg.norm(disp_ansys)
max_error = np.max(np.linalg.norm(disp_ansys - disp_pinn, axis=1))

print("===== PINN vs ANSYS =====")
print(f"L2 Relative Error : {l2_error:.4%}")
print(f"Max Error        : {max_error:.6e} m")

mapdl.exit()
