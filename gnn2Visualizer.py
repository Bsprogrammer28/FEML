import pyvista as pv
import numpy as np
import pandas as pd

# 1. LOAD DATA
nodes = np.load("Data\\Simulation_Data\\Simulation_Output\\fixed_node_coords.npy")    # (N, 3)
node_nums = np.load("Data\\Simulation_Data\\Simulation_Output\\fixed_node_nums.npy")  # (N,)
edge_index = np.load("Data\\Simulation_Data\\Simulation_Output\\GNN_Edge_Index_A.npy") # (2, E)
df_metadata = pd.read_csv("Data\\Simulation_Data\\Simulation_Output\\simulation_input_metadata.csv")

# 2. SELECT CASE FOR VISUALIZATION
case_idx = 0  # Change this to see different simulation cases
case_data = df_metadata.iloc[case_idx]
applied_node_id = int(case_data['Applied_Node_ID'])
load_mag = case_data['Load_Magnitude_N']

# Map ANSYS Node ID to array index
id_to_idx = {int(node_id): i for i, node_id in enumerate(node_nums)}
load_node_idx = id_to_idx.get(applied_node_id)

# 3. DEFINE NODE GROUPS (For coloring)
# Categories: 0=Normal, 1=Fixed, 2=Loaded
node_categories = np.zeros(nodes.shape[0])
fixed_mask = nodes[:, 0] < 0.001
node_categories[fixed_mask] = 1
if load_node_idx is not None:
    node_categories[load_node_idx] = 2

# 4. CREATE PYVISTA OBJECTS
# A. Point Cloud for Nodes
point_cloud = pv.PolyData(nodes)
point_cloud["Node Type"] = node_categories

# B. Lines for Edges
# Format: [2, start, end, 2, start, end...]
lines = np.empty((edge_index.shape[1], 3), dtype=np.int_)
lines[:, 0] = 2
lines[:, 1:] = edge_index.T
point_cloud.lines = lines

# C. Force Vector Arrow
if load_node_idx is not None:
    load_pos = nodes[load_node_idx]
    # Direction is FY negative
    direction = np.array([0, -1, 0]) 
    arrow = pv.Arrow(start=load_pos, direction=direction, scale=1.5)

# 5. PLOTTING
plotter = pv.Plotter(window_size=[1024, 768])
plotter.background_color = "white"

# Add the Mesh Edges
plotter.add_mesh(point_cloud, color="black", line_width=1, opacity=0.3, label="Mesh Edges")

# Add Nodes with categorical colors
# 0: lightgrey, 1: Red (Fixed), 2: Blue (Loaded)
cmap = ["lightgrey", "red", "blue"]
plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=8, 
                 scalars="Node Type", cmap=cmap, show_scalar_bar=False)

# Add Force Arrow
if load_node_idx is not None:
    plotter.add_mesh(arrow, color="blue", label=f"Load: {load_mag} N")
    plotter.add_point_labels([load_pos], [f"Node {applied_node_id}"], 
                             point_size=20, font_size=12, text_color="blue")

# Add Boundary Condition Label
plotter.add_text("X=0 Fixed Support", position=(20, 700), font_size=10, color="red")

# View Settings
plotter.add_legend(face='circle', bcolor=None)
plotter.camera_position = 'iso'
plotter.show()