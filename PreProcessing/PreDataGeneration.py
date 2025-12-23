import meshio
import numpy as np
import torch
import itertools

MSH_FILE = "D:\\Work\\Programming\\Machine Learning\\Python\\Projects\\FEML\\Data\\Geometry.msh"

FIXED_AXIS = 0          # 0=x, 1=y, 2=z
FIXED_TOL = 1e-6        # tolerance for BC detection

LOAD_NODE_ID = None     # set to None to auto-detect
LOAD_VECTOR = np.array([0.0, 0.0, -10000.0])  # Fx,Fy,Fz

mesh = meshio.read(MSH_FILE)

points = mesh.points.astype(np.float32)
cells = mesh.cells_dict

assert "tetra" in cells, "Only tetrahedral meshes supported"

elements = cells["tetra"]
num_nodes = points.shape[0]

edges = set()

for elem in elements:
    for i, j in itertools.combinations(elem, 2):
        edges.add((i, j))
        edges.add((j, i))   # undirected

edge_index = torch.tensor(list(edges), dtype=torch.long).t()

bc = np.zeros((num_nodes, 3), dtype=np.float32)

fixed_nodes = np.where(np.abs(points[:, FIXED_AXIS]) < FIXED_TOL)[0]
bc[fixed_nodes] = 1.0

loads = np.zeros((num_nodes, 3), dtype=np.float32)

if LOAD_NODE_ID is None:
    LOAD_NODE_ID = np.argmax(points[:, 2])  # highest Z node

loads[LOAD_NODE_ID] = LOAD_VECTOR

node_features = np.hstack([
    points,   # x,y,z
    bc,       # bc_x,bc_y,bc_z
    loads     # Fx,Fy,Fz
]).astype(np.float32)
dir = "D:\\Work\\Programming\\Machine Learning\\Python\\Projects\\FEML\\Data\\Automated"
np.save(f"{dir}\\node_features_X.npy", node_features)
np.save(f"{dir}\\edge_index_A.npy", edge_index.numpy())

meta_info = {
    "num_nodes": num_nodes,
    "num_edges": edge_index.shape[1],
    "num_elements": elements.shape[0],
    "element_type": "tetra",
    "fixed_axis": FIXED_AXIS,
    "load_node": int(LOAD_NODE_ID),
    "load_vector": LOAD_VECTOR.tolist()
}

np.save(f"{dir}\\meta_info.npy", meta_info) # type: ignore
print("✅ GNN files generated successfully")
