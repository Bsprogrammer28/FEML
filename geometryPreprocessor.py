import torch
from dataclasses import dataclass

def normalize_geometry(L, W, H):
    Lx = 1
    Ly = W / L
    Lz = H / L
    return Lx, Ly, Lz

def normalize_point(x, y, z, L, W, H):
    xn =  x / L
    yn = (y+ W/2) / L
    zn = (z+ H/2) / L
    return xn, yn, zn
# nn input = [0, 1]^3 

def gaussian_load(x, y, z, x0, y0, z0, F, sigma=0.02):
    return F * torch.exp(
        -((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))

@dataclass
class BeamPINNInput:
    # Scaling
    Lx: float  # Length in x direction
    Ly: float  # Length in y direction
    Lz: float  # Length in z direction

    # Material
    E: float   # Young's modulus
    nu: float  # Poisson's ratio

    # Load
    load_type: str
    direction: str
    magnitude: float
    location: tuple