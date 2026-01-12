import torch
import torchphysics as tp
import pytorch_lightning as pl

E = 210e9
nu = 0.3

mu = E / (2 * (1+nu))
lam = E * nu / ((1 + nu) * (1 - 2*nu))


# 3d Space
X = tp.spaces.R1('x')
Y = tp.spaces.R1('y')
Z = tp.spaces.R1('z')

U = tp.spaces.R1('u')  # deflection
V = tp.spaces.R1('v')  # deflection
W = tp.spaces.R1('w')  # deflection

Lx, Ly, Lz = 1, 0.2, 0.2

domain = (
    tp.domains.Interval(X, 0, Lx) *
    tp.domains.Interval(Y, 0, Ly) *
    tp.domains.Interval(Z, 0, Lz)
          )

model = tp.models.FCN(
    input_space=X*Y*Z,
    output_space=U*V*W,
    hidden=(128, 128, 128),
    activations=torch.nn.Tanh()
)


def strain(u, v, w, x, y, z):
    u_x = (1/Lx) * tp.utils.grad(u, x)
    u_y = (1/Ly) * tp.utils.grad(u, y)
    u_z = (1/Lz) * tp.utils.grad(u, z)

    v_x = (1/Lx) * tp.utils.grad(v, x)
    v_y = (1/Ly) * tp.utils.grad(v, y)
    v_z = (1/Lz) * tp.utils.grad(v, z)

    w_x = (1/Lx) * tp.utils.grad(w, x)
    w_y = (1/Ly) * tp.utils.grad(w, y)
    w_z = (1/Lz) * tp.utils.grad(w, z)

    eps_xx = u_x
    eps_yy = v_y
    eps_zz = w_z

    eps_xy = 0.5 * (u_y + v_x)
    eps_xz = 0.5 * (u_z + w_x)
    eps_yz = 0.5 * (v_z + w_y)

    return eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz

def stress(eps, lam, mu):
    eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz = eps
    trace = eps_xx + eps_yy + eps_zz

    s_xx = lam*trace + 2*mu*eps_xx
    s_yy = lam*trace + 2*mu*eps_yy
    s_zz = lam*trace + 2*mu*eps_zz
    s_xy = 2*mu*eps_xy
    s_xz = 2*mu*eps_xz
    s_yz = 2*mu*eps_yz

    return s_xx, s_yy, s_zz, s_xy, s_xz, s_yz

def elasticity_pde(u, v, w, x, y, z):
    eps = strain(u, v, w, x, y, z)
    s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = stress(eps, lam, mu)

    fx = tp.utils.grad(s_xx, x) + tp.utils.grad(s_xy, y) + tp.utils.grad(s_xz, z)
    fy = tp.utils.grad(s_xy, x) + tp.utils.grad(s_yy, y) + tp.utils.grad(s_yz, z)
    fz = tp.utils.grad(s_xz, x) + tp.utils.grad(s_yz, y) + tp.utils.grad(s_zz, z)
    scale = E/Lx

    return torch.cat([fx, fy, fz], dim=-1) / scale

def clamp(u, v, w):
    return torch.cat([u, v, w], dim=-1)


sampler_int = tp.samplers.RandomUniformSampler(domain, n_points=15_000)

# Left face (x = 0)
left_face_domain = tp.domains.Point(X, 0) * tp.domains.Interval(Y, 0, Ly) * tp.domains.Interval(Z, 0, Lz)

sampler_left = tp.samplers.RandomUniformSampler(
    left_face_domain,
    n_points=1_875
)
pde_cond = tp.conditions.PINNCondition(model, sampler_int, elasticity_pde)
bc_clamp = tp.conditions.PINNCondition(model, sampler_left, clamp)

bc_clamp.weight = 1e4
conds = [pde_cond, bc_clamp]

# Training
solver = tp.solver.Solver(
    conds,
    optimizer_setting=tp.OptimizerSetting(
        optimizer_class=torch.optim.Adam,
        lr=1e-4
    )
)

trainer = pl.Trainer(
    max_steps=15000,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(solver)

# Switch optimizer to LBFGS
solver.optimizer = torch.optim.LBFGS(
    solver.parameters(),
    lr=1.0,
    max_iter=500,
    tolerance_grad=1e-9,
    tolerance_change=1e-9,
    history_size=50
)

# Continue training with LBFGS
trainer = pl.Trainer(
    max_steps=500,   
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
)

trainer.fit(solver)


torch.save(model.state_dict(), "beam_model3d.pth")
print("Training complete. Model saved to 'beam_model3d.pth'.")