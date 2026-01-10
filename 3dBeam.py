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
