import torch
import torchphysics as tp
import pytorch_lightning as pl

# Parameterization
Z = tp.spaces.R1('z')  # normalized 0 to 1
L = tp.spaces.R1('l')  # length
A = tp.spaces.R1('a')  # relative load posn
F = tp.spaces.R1('f')  # load magnitude

U = tp.spaces.R1('u')  # deflection

space_interval = tp.domains.Interval(Z, 0, 1)
param_interval = tp.domains.Interval(
    L, 1, 5) * tp.domains.Interval(A, 0.1, 1) * tp.domains.Interval(F, -100, 100)

full_domain = space_interval * param_interval

model = tp.models.FCN(
    input_space=Z*L*A*F,
    output_space=U,
    hidden=(128, 128, 128, 128),
    activations=torch.nn.Tanh()
)


def pde_residual(u, z, l, a, f):
    u_z = tp.utils.grad(u, z)
    u_zz = tp.utils.grad(u_z, z)
    u_zzz = tp.utils.grad(u_zz, z)
    u_zzzz = tp.utils.grad(u_zzz, z)

    u_xxxx = u_zzzz / (l**4)
    sigma = 200
    load_distribution = f * torch.exp(-sigma * (z - a)**2)

    return u_xxxx - load_distribution


def clamped_bc(u): return u

def clamped_slope(u, z, l):
    return tp.utils.grad(u, z) / l

def free_moment(u, z, l):
    return tp.utils.grad(tp.utils.grad(u, z), z)  / l**2

def free_shear(u, z, l):
    return tp.utils.grad(tp.utils.grad(tp.utils.grad(u, z), z), z) / l**3


sampler_pde = tp.samplers.RandomUniformSampler(full_domain, n_points=50000)

boundary_left = space_interval.boundary_left * param_interval
boundary_right = space_interval.boundary_right * param_interval

sampler_left  = tp.samplers.RandomUniformSampler(boundary_left, 8000)
sampler_right = tp.samplers.RandomUniformSampler(boundary_right, 8000)

pde_cond = tp.conditions.PINNCondition(model, sampler_pde, pde_residual)
bc1 = tp.conditions.PINNCondition(model, sampler_left, clamped_bc)
bc2 = tp.conditions.PINNCondition(model, sampler_left, clamped_slope)
bc3 = tp.conditions.PINNCondition(model, sampler_right, free_moment)
bc4 = tp.conditions.PINNCondition(model, sampler_right, free_shear)

bc1.weight = 20
bc2.weight = 20

conds = [pde_cond, bc1, bc2, bc3, bc4]

# Training

solver = tp.solver.Solver(
    conds,
    optimizer_setting=tp.OptimizerSetting(
        optimizer_class=torch.optim.Adam,
        lr=1e-3
    )
)

trainer = pl.Trainer(
    max_steps=25000,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(solver)

torch.save(model.state_dict(), "beam_modelv2.pth")
print("Training complete. Model saved to 'beam_modelv2.pth'.")