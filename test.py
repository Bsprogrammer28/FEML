import pyvista as pv

W, H = 0.2, 0.2

clamp = pv.Plane(
    center=(-0.1, 0, 0),
    direction=(1, 0, 0),
    i_size=W,
    j_size=H
)

clamp.plot(show_edges=True)
