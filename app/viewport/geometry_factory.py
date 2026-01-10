import pyvista as pv

def create_beam(L, W, H):
    beam = pv.Box(
        bounds=(0, L,
                -W / 2, W / 2,
                -H / 2, H / 2)
    )
    return beam

def create_clamp_face(W, H):
    face = pv.Plane(
        center=(1e-3, 0, 0),
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