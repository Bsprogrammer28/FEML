import pyvista as pv
import numpy as np

# Load Graph Components
nodes = np.load("Data\\Simulation_Data\\Simulation_Output\\GNN_Node_Features_X.npy") # The (x, y, z) coordinates
edge_index = np.load("Data\\Simulation_Data\\Simulation_Output\\GNN_Edge_Index_A.npy") # The (2, E) edge list

# 1. Create the points
point_cloud = pv.PolyData(nodes)

# 2. Create the lines (edges) for the graph
# PyVista lines format: [2, start_node, end_node, 2, start_node, end_node, ...]
lines = np.empty((edge_index.shape[1], 3), dtype=np.int_)
lines[:, 0] = 2
lines[:, 1:] = edge_index.T
graph_edges = pv.PolyData(nodes)
graph_edges.lines = lines

# 3. Plot the Graph
plotter = pv.Plotter()
# Add nodes as spheres
plotter.add_mesh(point_cloud, render_points_as_spheres=True, color='red', point_size=5)
# Add edges as lines
plotter.add_mesh(graph_edges, color='black', line_width=1)
plotter.add_text("GNN Graph Representation (Nodes & Edges)", font_size=10)
plotter.show()