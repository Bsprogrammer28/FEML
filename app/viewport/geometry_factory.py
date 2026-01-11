import pyvista as pv
import numpy as np

def create_beam(L, W, H):
    beam = pv.Box(
        bounds=(0, L,
                -W / 2, W / 2,
                -H / 2, H / 2)
    )
    return beam

def create_clamp_face(W, H):
    face = pv.Plane(
        center=(0, 0, 0),
        direction=(1, 0, 0),
        i_size=W,
        j_size=H
    )
    return face

def create_force_arrow(origin, direction, scale=1.0):
    arrow = pv.Arrow(
        start=origin,
        direction=direction,
        scale=scale
    )
    return arrow

def sample_mesh(mesh, n_points=5000):
    surf = mesh.extract_surface().triangulate()
    pts = surf.points

    if pts.shape[0] > n_points:
        idx = np.random.choice(pts.shape[0], n_points, replace=False)
        pts = pts[idx]

        surf = surf.extract_points(idx, adjacent_cells=True)
    return surf