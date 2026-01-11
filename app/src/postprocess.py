import numpy as np

def fake_displacement(points, L, magnitude):
    x = points[:, 0]
    w = magnitude*(x/L)**2 * (3 - 2*(x/L))
    disp = np.zeros_like(points)
    disp[:, 1] = w
    return disp