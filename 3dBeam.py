import torch
import torchphysics as tp
import pytorch_lightning as pl

# 3d Space
X = tp.spaces.R1('x')
Y = tp.spaces.R1('y')
Z = tp.spaces.R1('z')

U = tp.spaces.R1('u')  # deflection
V = tp.spaces.R1('v')  # deflection
W = tp.spaces.R1('w')  # deflection

Lx, Ly, Lz = 1, 0.2, 0.2

domain = (tp.domains.Interval(X, 0, Lx) *
          tp.domains.Interval(Y, 0, Ly) *
          tp.domains.Interval(Z, 0, Lz))
model = tp.models.FCN(
    input_space=X*Y*Z,
    output_space=U*V*W,
    hidden=(256, 256, 256, 256),
    activations=torch.nn.Tanh()
)


def strain(u, v, w, x, y, z):
    u_x = tp.utils.grad(u, x)
    u_y = tp.utils.grad(u, y)
    u_z = tp.utils.grad(u, z)

    v_x = tp.utils.grad(v, x)
    v_y = tp.utils.grad(v, y)
    v_z = tp.utils.grad(v, z)

    w_x = tp.utils.grad(w, x)
    w_y = tp.utils.grad(w, y)
    w_z = tp.utils.grad(w, z)

    eps_xx = u_x
    eps_yy = v_y
    eps_zz = w_z

    eps_xy = 0.5 * (u_y + v_x)
    eps_xz = 0.5 * (u_z + w_x)
    eps_yz = 0.5 * (v_z + w_y)

    return eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz

def stress(eps, lam, mu):
    eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz = eps
    trace = eps_xx, eps_yy, eps_zz

    s_xx = lam*trace + 2*mu*eps_xx
    s_yy = lam*trace + 2*mu*eps_yy
    s_zz = lam*trace + 2*mu*eps_zz
    s_xy = 2*mu*eps_xy
    s_xz = 2*mu*eps_xz
    s_yz = 2*mu*eps_yz

    return s_xx, eps_yy, eps_zz, s_xy, s_xz, s_yz

def elasticity_pde(u, v, w, x, y, z):
    eps = strain(u, v, w, x, y, z)
    s_xx, s_yy, s_zz, 
