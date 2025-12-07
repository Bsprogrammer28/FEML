import os
import numpy as np
import pandas as pd
import pyvista as pv

# -----------------------------
# 1. Paths
# -----------------------------
base_output_dir = "Data\\Simulation_Data"
output_dir = os.path.join(base_output_dir, "Simulation_Output")

coords_path = os.path.join(output_dir, "fixed_node_coords.npy")
nodes_path = os.path.join(output_dir, "fixed_node_nums.npy")
disp_path = os.path.join(output_dir, "all_displacement_tensors.npy")
csv_path = os.path.join(output_dir, "simulation_input_metadata.csv")

# -----------------------------
# 2. Load data
# -----------------------------
node_coords = np.load(coords_path)          # (N_nodes, 3)
node_nums   = np.load(nodes_path)           # (N_nodes,)
all_displacements = np.load(disp_path)      # (N_cases, N_nodes, 3)
meta = pd.read_csv(csv_path)

n_cases, n_nodes, n_comp = all_displacements.shape
print(f"Loaded {n_cases} cases, {n_nodes} nodes, {n_comp} comps.")

# -----------------------------
# 3. Pick a case to visualize
# -----------------------------
# Case with maximum deformation (nice looking)
case_idx = int(meta["Max_Deformation_m"].idxmax())
case_meta = meta.iloc[case_idx]

print("\nVisualizing case:")
print(f"  Case index (0-based): {case_idx}")
print(f"  Case_ID             : {case_meta['Case_ID']}")
print(f"  Load_Position_X_m   : {case_meta['Load_Position_X_m']}")
print(f"  Load_Magnitude_N    : {case_meta['Load_Magnitude_N']}")
print(f"  Applied_Node_ID     : {case_meta['Applied_Node_ID']}")
print(f"  Max_Deformation_m   : {case_meta['Max_Deformation_m']:.3e}")

disp = all_displacements[case_idx]  # (N_nodes, 3)
disp_mag = np.linalg.norm(disp, axis=1)

# -----------------------------
# 4. Build a volumetric mesh via 3D Delaunay
#    (no need for element connectivity)
# -----------------------------
points = node_coords.copy()

# Create a point cloud with displacement as point data
cloud = pv.PolyData(points)
cloud["disp"] = disp
cloud["disp_mag"] = disp_mag

# Delaunay 3D to generate tetrahedral volume
print("\nRunning 3D Delaunay triangulation for visualization mesh...")
volume = cloud.delaunay_3d()         # tetra mesh
surface = volume.extract_surface()   # outer solid surface

print("Created volume with", volume.n_cells, "tets and", volume.n_points, "points.")

# -----------------------------
# 5. Warp by displacement (deformed geometry)
# -----------------------------
SCALE = 50.0  # scale factor for deformation visualization
volume_warped = volume.warp_by_vector("disp", factor=SCALE)
surface_warped = volume_warped.extract_surface()

# -----------------------------
# 6. Identify BC nodes & load node in warped space
# -----------------------------
BC_TOL = 1e-3
fixed_mask = node_coords[:, 0] <= BC_TOL

# undeformed -> warped: x_def = x + SCALE*u
warped_points_full = node_coords + SCALE * disp

fixed_points_warped = warped_points_full[fixed_mask]

applied_node_id = int(case_meta["Applied_Node_ID"])
matches = np.where(node_nums == applied_node_id)[0]
if len(matches) == 0:
    raise ValueError(f"Applied node ID {applied_node_id} not found in node_nums.")
load_idx = int(matches[0])
load_point_warped = warped_points_full[load_idx]

# BC points + load arrow as separate geometries
bc_points_pd = pv.PolyData(fixed_points_warped)

force_dir = np.array([0.0, -1.0, 0.0])  # FY negative
force_arrow_len = 0.5                   # tweak this if needed
arrow = pv.Arrow(start=load_point_warped,
                 direction=force_dir,
                 scale=force_arrow_len)

# -----------------------------
# 7. Plot: undeformed wireframe + deformed solid + BC + force
# -----------------------------
plotter = pv.Plotter()

# Undeformed outline (wireframe from Delaunay surface)
plotter.add_mesh(
    surface,
    style="wireframe",
    opacity=0.2,
    color="white",
    label="Undeformed"
)

# Deformed solid surface with displacement contour
plotter.add_mesh(
    surface_warped,
    scalars="disp_mag",
    cmap="viridis",
    show_edges=False,
    opacity=1.0,
    label="Deformed (scaled)"
)

# BC nodes (fixed end) as red spheres
plotter.add_mesh(
    bc_points_pd,
    color="red",
    point_size=12,
    render_points_as_spheres=True,
    label="Fixed BC"
)

# Load arrow at applied node
plotter.add_mesh(
    arrow,
    color="yellow",
    label="Applied force"
)

plotter.add_scalar_bar(title="|u| [m]", n_labels=5)
plotter.add_axes()
plotter.show_bounds(grid="back", location="outer")
plotter.add_legend()
plotter.view_isometric()

plotter.show(title="Solid deformation (Delaunay volume, BC & force)")
