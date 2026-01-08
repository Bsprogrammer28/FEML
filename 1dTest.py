import torch
import torchphysics as tp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision('medium')

Z = tp.spaces.R1('z')
L = tp.spaces.R1('l')
A = tp.spaces.R1('a')
F = tp.spaces.R1('f')
U = tp.spaces.R1('u')

space_interval = tp.domains.Interval(Z, 0, 1)
param_interval = tp.domains.Interval(L, 1, 5) * tp.domains.Interval(A, 0.1, 1) * tp.domains.Interval(F, -100, 100)
full_domain = space_interval * param_interval

model = tp.models.FCN(
    input_space=Z*L*A*F,
    output_space=U,
    hidden=(64, 64, 64, 64, 64),
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
def clamped_slope(u, z, l): return tp.utils.grad(u, z) / l
def free_moment(u, z, l): return tp.utils.grad(tp.utils.grad(u, z), z) / l**2
def free_shear(u, z, l): return tp.utils.grad(tp.utils.grad(tp.utils.grad(u, z), z), z) / l**3

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

bc1.weight = 20.0
bc2.weight = 20.0
conds = [pde_cond, bc1, bc2, bc3, bc4]

# FIX: Switched to StepLR. 
# It decays LR by half every 5000 steps. No 'monitor' required.
optimizer = tp.OptimizerSetting(
    optimizer_class=torch.optim.AdamW,
    lr=2e-3,
    optimizer_args={'weight_decay': 1e-5},
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_args={'step_size': 5000, 'gamma': 0.5},
    scheduler_frequency=1
)

solver = tp.solver.Solver(conds, optimizer_setting=optimizer)

class ConsoleLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 100 == 0:
            loss = trainer.callback_metrics.get("val_loss", trainer.callback_metrics.get("train_loss"))
            print(f"Epoch {trainer.current_epoch}: Loss = {loss:.6f}")

trainer = pl.Trainer(
    max_steps=25000,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    callbacks=[ConsoleLogger()],
    check_val_every_n_epoch=1000
)

print("Starting training on:", trainer.device_ids if trainer.device_ids else "CPU")
trainer.fit(solver)

torch.save(model.state_dict(), "beam_model_optimized.pth")
print("Training complete. Model saved to 'beam_model_optimized.pth'.")