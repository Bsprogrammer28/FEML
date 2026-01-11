import torch
import torchphysics as tp
import pytorch_lightning as pl
import os
import numpy as np
import matplotlib.pyplot as plt


E = 210e9
nu = 0.3

mu = E / (2 * (1+nu))
lam = E * nu / ((1 + nu) * (1 - 2*nu))


# 3d Space
X = tp.spaces.R1('x')
Y = tp.spaces.R1('y')
Z = tp.spaces.R1('z')

Fx = tp.spaces.R1('fx')
Fy = tp.spaces.R1('fy')
Fz = tp.spaces.R1('fz')

U = tp.spaces.R1('u')  # deflection
V = tp.spaces.R1('v')  # deflection
W = tp.spaces.R1('w')  # deflection

Lx, Ly, Lz = 1, 0.2, 0.2

domain = (
    tp.domains.Interval(X, 0, Lx) *
    tp.domains.Interval(Y, 0, Ly) *
    tp.domains.Interval(Z, 0, Lz) *
    tp.domains.Interval(Fx, -1, 1) *
    tp.domains.Interval(Fy, -1, 1) *
    tp.domains.Interval(Fz, -1, 1)
)

model = tp.models.FCN(
    input_space=X*Y*Z*Fx*Fy*Fz,
    output_space=U*V*W,
    hidden=(128, 128, 128),
    activations=torch.nn.Tanh()
)

def traction_bc(u, v, w, x, y, z, fx, fy, fz):
    eps = strain(u, v, w, x, y, z)
    s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = stress(eps, lam, mu)

    tx = s_xx - fx
    ty = s_xy - fy
    tz = s_xz - fz

    return torch.cat([tx, ty, tz], dim=-1)

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


def elasticity_pde(u, v, w, x, y, z, fx, fy, fz):
    eps = strain(u, v, w, x, y, z)
    s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = stress(eps, lam, mu)

    fx_int = (
        tp.utils.grad(s_xx, x)
        + tp.utils.grad(s_xy, y)
        + tp.utils.grad(s_xz, z)
    )
    fy_int = (
        tp.utils.grad(s_xy, x)
        + tp.utils.grad(s_yy, y)
        + tp.utils.grad(s_yz, z)
    )
    fz_int = (
        tp.utils.grad(s_xz, x)
        + tp.utils.grad(s_yz, y)
        + tp.utils.grad(s_zz, z)
    )

    scale = E / Lx
    sigma_ref = E  # reference stress scale

    res = torch.cat([fx_int, fy_int, fz_int], dim=-1)
    return res / sigma_ref



def clamp(u, v, w, x, y, z, fx, fy, fz):
    return torch.cat([u, v, w], dim=-1)


sampler_int = tp.samplers.RandomUniformSampler(domain, n_points=15_000)

# Left face (x = 0)
left_face_domain = (
    tp.domains.Point(X, 0)
    * tp.domains.Interval(Y, 0, Ly)
    * tp.domains.Interval(Z, 0, Lz)
    * tp.domains.Point(Fx, 0.0)
    * tp.domains.Point(Fy, 0.0)
    * tp.domains.Point(Fz, 0.0)
)

sampler_left = tp.samplers.RandomUniformSampler(
    left_face_domain,
    n_points=1_875
)

right_face = (
    tp.domains.Point(X, Lx) *
    tp.domains.Interval(Y, 0, Ly) *
    tp.domains.Interval(Z, 0, Lz) *
    tp.domains.Interval(Fx, -1, 1) *
    tp.domains.Interval(Fy, -1, 1) *
    tp.domains.Interval(Fz, -1, 1)
)

sampler_right = tp.samplers.RandomUniformSampler(right_face, 2000)

pde_cond = tp.conditions.PINNCondition(model, sampler_int, elasticity_pde)
bc_clamp = tp.conditions.PINNCondition(model, sampler_left, clamp)
bc_traction = tp.conditions.PINNCondition(
    model, sampler_right, traction_bc
)

bc_traction.weight = 10
bc_clamp.weight = 1e4
conds = [pde_cond, bc_clamp, bc_traction]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
solver = tp.solver.Solver(
    conds,
    optimizer_setting=tp.OptimizerSetting(
        optimizer_class=torch.optim.Adam,
        lr=1e-4
    )
)

os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/history", exist_ok=True)

history = {
    "iter": [],
    "loss": []
}

start_iter = 0

checkpoint_path = "results/checkpoints/latest.pt"

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    history = ckpt["history"]
    start_iter = ckpt["iter"]

trainer = pl.Trainer(
    max_steps=start_iter + 25000,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(solver)

metrics = trainer.callback_metrics

if "train_loss" in metrics:
    loss_val = metrics["train_loss"]
elif "loss_total" in metrics:
    loss_val = metrics["loss_total"]
else:
    loss_val = list(metrics.values())[0]  # fallback

history["iter"].append(trainer.global_step)
history["loss"].append(loss_val.item())


torch.save(
    {
        "iter": trainer.global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history
    },
    checkpoint_path
)

np.save("results/history/training_history.npy", history)

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

torch.save(model.state_dict(), "models/beam_model3d.pth")
print("Training complete. Model saved to 'beam_model3d.pth'.")

plt.figure()
plt.semilogy(history["iter"], history["loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()