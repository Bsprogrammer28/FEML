"""
Generate GNN input files from FEA exports.

Produces:
 - GNN_Node_Features_X.npy  (node coordinates, shape (N_nodes,3))
 - GNN_Edge_Index_A.npy     (edge_index shape (2, E))

This script reads the files saved by `dataGen.py` in
`Data\Simulation_Data\Simulation_Output` and converts the element
connectivity into an edge index suitable for PyTorch-Geometric.

Usage (from repo root):
    python scripts\generate_gnn_files.py

Notes:
 - `fixed_node_nums.npy` contains MAPDL node numbers. Edge indices
   are returned as 0-based indices that align with the order of
   rows in `fixed_node_coords.npy`.
 - The script emits undirected edges as bidirectional pairs
   (i->j and j->i) to match PyG expectations.
"""

import os
import numpy as np
from itertools import combinations


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_inputs(output_dir):
    coords_p = os.path.join(output_dir, "fixed_node_coords.npy")
    nums_p = os.path.join(output_dir, "fixed_node_nums.npy")
    conn_p = os.path.join(output_dir, "fixed_connectivity.npy")

    X = np.load(coords_p)
    node_nums = np.load(nums_p)
    conn = np.load(conn_p, allow_pickle=True)
    return X, node_nums, conn


def connectivity_to_edge_index(connectivity, node_num_to_idx):
    """
    connectivity: array-like; each element row contains node numbers
    node_num_to_idx: dict mapping node number -> 0-based index

    Returns edge_index as (2, E) numpy array with bidirectional edges.
    """
    edges = set()

    # Normalize connectivity to a list of lists
    if connectivity.ndim == 2:
        elems = connectivity.tolist()
    else:
        # object array (ragged) -> convert
        elems = [np.asarray(row).tolist() for row in connectivity]

    for el_nodes in elems:
        # remove NaNs or zero-length entries
        el_nodes = [int(n) for n in el_nodes if n is not None]
        if len(el_nodes) < 2:
            continue
        for a, b in combinations(el_nodes, 2):
            if a not in node_num_to_idx or b not in node_num_to_idx:
                # skip if mapping not found (shouldn't happen in consistent exports)
                continue
            ia = node_num_to_idx[a]
            ib = node_num_to_idx[b]
            if ia == ib:
                continue
            # store undirected edge as ordered tuple (min, max)
            if ia < ib:
                edges.add((ia, ib))
            else:
                edges.add((ib, ia))

    # Expand to bidirectional
    src = []
    dst = []
    for a, b in sorted(edges):
        src.append(a); dst.append(b)
        src.append(b); dst.append(a)

    edge_index = np.vstack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)])
    return edge_index


def main():
    base_output_dir = os.path.join("Data", "Simulation_Data")
    output_dir = os.path.join(base_output_dir, "Simulation_Output")

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Expected output directory not found: {output_dir}")

    X, node_nums, conn = load_inputs(output_dir)

    print(f"Loaded coords: {X.shape}, node_nums: {node_nums.shape}")

    # Save node features as-is
    gnn_X_path = os.path.join(output_dir, "GNN_Node_Features_X.npy")
    np.save(gnn_X_path, X)
    print(f"Saved GNN node features to: {gnn_X_path}")

    # Build mapping: MAPDL node number -> 0-based index in coords array
    node_num_to_idx = {int(num): idx for idx, num in enumerate(node_nums)}

    print("Converting connectivity to edge_index (this may take a moment)...")
    edge_index = connectivity_to_edge_index(conn, node_num_to_idx)

    gnn_edge_path = os.path.join(output_dir, "GNN_Edge_Index_A.npy")
    np.save(gnn_edge_path, edge_index)
    print(f"Saved GNN edge index to: {gnn_edge_path} (shape: {edge_index.shape})")


if __name__ == "__main__":
    main()
