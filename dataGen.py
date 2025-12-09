# ==========================================
# 0. CONSTANTS & IMPORTS
# ==========================================
LENGTH = 10.0          # meters
WIDTH = 1.0            # meters
HEIGHT = 1.0           # meters
DENSITY = 7850         # kg/m^3
YOUNG_MODULUS = 200e9  # Pascals
POISSON_RATIO = 0.3
ESIZE = 0.5            # Element size for meshing (m)

import numpy as np
import pandas as pd
import os
from ansys.mapdl.core import launch_mapdl

# ==========================================
# 1. CONFIGURATION & LOAD CASE GENERATION
# ==========================================
base_output_dir = "Data\\Simulation_Data"
output_dir = os.path.join(base_output_dir, "Simulation_Output")
os.makedirs(output_dir, exist_ok=True)

POSITIONS_STEPS = 1000
FORCES_STEPS = 10

positions = np.linspace(0.01, LENGTH, POSITIONS_STEPS)
forces = np.linspace(100, 10000, FORCES_STEPS)

cases = []
case_counter = 0
for x_loc in positions:
    for mag in forces:
        case_counter += 1
        cases.append({
            "id": case_counter,
            "x_loc": round(x_loc, 3),
            "force": mag
        })
print(f"Generated {len(cases)} load cases to simulate.")

# Storage
displacement_tensors = []
results_data = []

# ==========================================
# 2. SETUP ANSYS (Geometry, Mesh, BCs)
# ==========================================
mapdl = launch_mapdl(run_location=base_output_dir,
                     loglevel="ERROR",
                     override=True)

mapdl.clear()
mapdl.prep7()

# 2a. Material
mapdl.mp("DENS", 1, DENSITY)
mapdl.mp("EX", 1, YOUNG_MODULUS)
mapdl.mp("PRXY", 1, POISSON_RATIO)

# 2b. Geometry (cantilever block)
mapdl.et(1, "SOLID186")
mapdl.block(0, LENGTH,
            -WIDTH / 2,  WIDTH / 2,
            -HEIGHT / 2, HEIGHT / 2)

# 2c. Meshing
mapdl.lesize("ALL", ESIZE)
mapdl.vmesh("ALL")
mapdl.emodif("ALL", "MAT", 1)

# 2d. Fixed support at X=0
TOLERANCE_FOR_BC = 0.001
mapdl.nsel("S", "LOC", "X", -TOLERANCE_FOR_BC, TOLERANCE_FOR_BC)
mapdl.d("ALL", "ALL")
mapdl.allsel()

# 2e. Load application region (top front corner band)
TARGET_Y = WIDTH / 2.0
TARGET_Z = HEIGHT / 2.0
LOAD_TOL = ESIZE / 2.0

# ==========================================
# 3. SAVE FIXED MESH DATA
# ==========================================
node_coords = mapdl.mesh.nodes
node_nums = mapdl.mesh.nnum
connectivity = mapdl.mesh.elem

np.save(os.path.join(output_dir, "fixed_node_coords.npy"), node_coords)
np.save(os.path.join(output_dir, "fixed_node_nums.npy"), node_nums)
np.save(os.path.join(output_dir, "fixed_connectivity.npy"), connectivity)

print(
    f"Mesh data extracted. Total nodes: {len(node_nums)}. "
    f"Saved coordinates and connectivity."
)

# ==========================================
# 4. SIMULATION LOOP
# ==========================================

valid_cases = 0

for idx, case in enumerate(cases):
    c_id = case["id"]
    x_loc = case["x_loc"]
    f_mag = case["force"]

    # ---- SOLUTION PHASE ----
    mapdl.finish()
    mapdl.run("/SOLU")
    if idx == 0:
        mapdl.antype("STATIC")  # set analysis type once (or every time, both fine)

    # A. Remove previous forces
    mapdl.fdele("ALL", "ALL")

    # B. Select node(s) near target location on top surface
    mapdl.nsel("S", "LOC", "X", x_loc - LOAD_TOL, x_loc + LOAD_TOL)
    mapdl.nsel("R", "LOC", "Y", TARGET_Y - LOAD_TOL, TARGET_Y + LOAD_TOL)
    mapdl.nsel("R", "LOC", "Z", TARGET_Z - LOAD_TOL, TARGET_Z + LOAD_TOL)

    ncount = int(mapdl.get_value("NODE", 0, "COUNT"))

    if c_id <= 5:
        print(f"[DEBUG] Case {c_id}: x_loc={x_loc:.3f}, selected nodes={ncount}")

    if ncount == 0:
        mapdl.allsel()
        continue

    load_node_id = int(mapdl.get_value("NODE", 0, "NUM", "MAX"))

    if c_id <= 5:
        print(f"[DEBUG] Case {c_id}: using node {load_node_id} for load")

    # C. Apply force
    mapdl.f(load_node_id, "FY", -f_mag)

    # D. Solve
    mapdl.allsel()
    sol_out = mapdl.solve()

    # ---- POST-PROCESSING PHASE ----
    mapdl.finish()
    mapdl.post1()
    mapdl.set("LAST")

    # E. Displacements
    displacements = mapdl.post_processing.nodal_displacement("ALL")
    displacement_tensors.append(displacements)

    # F. Max deformation magnitude (for logging)
    mapdl.nsort("U", "SUM")
    max_def = mapdl.get_value("SORT", 0, "MAX")

    results_data.append({
        "Case_ID": c_id,
        "Load_Position_X_m": x_loc,
        "Applied_Node_ID": load_node_id,
        "Load_Magnitude_N": f_mag,
        "Max_Deformation_m": max_def,
    })

    valid_cases += 1

    if c_id % 200 == 0 or c_id == 1:
        print(
            f"Case {c_id}: X={x_loc:.3f} m, "
            f"F={f_mag:.1f} N → Max Def={max_def:.3e} m. "
            f"Valid cases so far={valid_cases}"
        )

print(f"\nTotal valid simulated cases = {valid_cases}")

# ==========================================
# 5. EXPORT
# ==========================================
if displacement_tensors:
    all_displacements = np.array(displacement_tensors)
    np.save(
        os.path.join(output_dir, "all_displacement_tensors.npy"),
        all_displacements
    )
    print(
        "\n--- SUCCESS ---\n"
        f"Saved ALL nodal displacement data to 'all_displacement_tensors.npy' "
        f"(Shape: {all_displacements.shape})"
    )
else:
    print(
        "\nWARNING: No displacement data was collected. "
        "Check node selection / loads."
    )

print(f"len(results_data) = {len(results_data)}")

df_out = pd.DataFrame(results_data)
csv_path = os.path.join(output_dir, "simulation_input_metadata.csv")
df_out.to_csv(csv_path, index=False)
print(f"Input metadata saved to {csv_path}")

mapdl.exit()
