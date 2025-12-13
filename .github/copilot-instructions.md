**Repository Overview**

- **Purpose**: FEML builds a surrogate GNN model that predicts FEA nodal displacements from boundary/load inputs. The repo contains scripts to generate FEA cases (`dataGen.py`), verify outputs (`verifyData.py`), convert FEA outputs into graph datasets (`meshToGraph.py`), and visualize results (`visualization.py`).

**Key Files**
- `dataGen.py`: Launches MAPDL (Ansys MAPDL) to build a mesh, run many static load cases, and save outputs to `Data\Simulation_Data\Simulation_Output`. Produces `fixed_node_coords.npy`, `fixed_node_nums.npy`, `fixed_connectivity.npy`, `all_displacement_tensors.npy`, and `simulation_input_metadata.csv`.
- `verifyData.py`: Loads the saved arrays and checks shape consistency (use this first to confirm `dataGen.py` outputs).
- `meshToGraph.py`: Builds PyTorch-Geometric `Data` objects for each case. Expects node features and edge index files: `GNN_Node_Features_X.npy` (node coords shape `(N_nodes,3)`) and `GNN_Edge_Index_A.npy` (edge_index shape `(2, E)`). Creates per-case features [x,y,z,Load_Position_X_m,Load_Magnitude_N] and targets `all_displacement_tensors.npy` shaped `(N_cases, N_nodes, 3)`.
- `visualization.py`: Uses `pyvista` to render undeformed + deformed geometry. Reads the same `Simulation_Output` files and shows BCs and applied force arrow.

**Data layout & shapes (explicit)**
- `fixed_node_coords.npy`: (N_nodes, 3) — node coordinates (x,y,z).
- `fixed_node_nums.npy`: (N_nodes,) — original node numbers from MAPDL.
- `fixed_connectivity.npy`: element connectivity (used for reference; some visualizations build a Delaunay volume instead).
- `all_displacement_tensors.npy`: (N_cases, N_nodes, 3).
- `simulation_input_metadata.csv`: per-case metadata with columns including `Case_ID`, `Load_Position_X_m`, `Load_Magnitude_N`, `Applied_Node_ID`, `Max_Deformation_m`.

**How components interact (big picture)**
- `dataGen.py` (FEA) → writes arrays in `Data\Simulation_Data\Simulation_Output`.
- `verifyData.py` checks the outputs for consistent shapes.
- `meshToGraph.py` consumes the saved arrays (or preprocessed `GNN_*` files) and constructs a PyG dataset used to define and train the `EncoderProcessorDecoderGNN` model (in-file model class).
- `visualization.py` reads outputs and shows an example case (max deformation).

**Concrete run examples (Windows `cmd.exe`)**
- Create env and install dependencies (recommended):
```
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas pyvista torch torchvision torchaudio
# torch-geometric has special install commands depending on CUDA; see https://pytorch-geometric.readthedocs.io
pip install ansys-mapdl-core
```
- Generate data (requires Ansys MAPDL installed & licensed):
```
python dataGen.py
```
- Verify outputs:
```
python verifyData.py
```
- Build dataset / test model pipeline (reads/expects `GNN_*` files):
```
python meshToGraph.py
```
- Visualize a case:
```
python visualization.py
```

**Project-specific conventions & gotchas**
- Paths: scripts use relative Windows-style paths like `Data\\Simulation_Data`. When running from a different CWD, run scripts from repository root or update path constants at top of files.
- Large scale: `dataGen.py` is configured to generate many cases (default 1000 positions × 10 force magnitudes → 10k cases). This is computationally and time intensive — only run full sweep on machines with MAPDL & adequate compute/time.
- Preprocessed GNN files: `meshToGraph.py` expects `GNN_Node_Features_X.npy` and `GNN_Edge_Index_A.npy`. If those are missing, the script falls back to mocking targets but will not produce a trainable dataset. `GNN_Node_Features_X.npy` is effectively `fixed_node_coords.npy` (shape `(N_nodes,3)`) and `GNN_Edge_Index_A.npy` should be an edge index array of shape `(2, E)`. If you need to create them from saved connectivity, convert connectivity to an edge index (unique undirected edges) and save with `np.save`.
- Model conventions: The in-repo model class is `EncoderProcessorDecoderGNN` in `meshToGraph.py`. It expects `in_channels=5` (x,y,z,Lx,Fy) and outputs 3 DOFs (dx,dy,dz). Default hyperparameters in the file: `HIDDEN_DIM=64`, `GNN_LAYERS=4`, `batch_size=32` in DataLoader examples.

**Dependencies (detectable from code)**
- Mandatory: `numpy`, `pandas`.
- FEA integration: `ansys-mapdl-core` (MAPDL must be installed and licensed).
- ML: `torch`, `torch_geometric` (PyG) — note PyG installs differ by CUDA/CPU.
- Visualization: `pyvista`.

**When you are the agent: priorities for edits**
- Preserve the MAPDL launch flow and output names in `dataGen.py` — downstream scripts expect those exact filenames.
- When editing `meshToGraph.py` focus on (1) making `GNN_*` file creation explicit (small helper to convert `fixed_*` to `GNN_*`), (2) moving the model to `modelArchitecture.py` and adding a runnable `train()` in `traning.py` (there is a commented training loop in `meshToGraph.py`).

**If anything is unclear or missing**
- Tell me which workflow you want prioritized (FEA runs, dataset creation, or model training). Also confirm whether you want me to add a `requirements.txt` and a short `setup` script to generate `GNN_*` files from `fixed_*` arrays — I can add those next.