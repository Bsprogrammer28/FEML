import deepxde as dde

# Define the PDE (Euler-Bernoulli)
def pde(x, y):
    dy_xxxx = dde.grad.hessian(y, x, i=0, j=0) # simplified 4th deriv
    return dy_xxxx - load_constant 

# Define Geometry and Boundaries
geom = dde.geometry.Interval(0, 1)
bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_check, component=0)

# Build and Train
data = dde.data.PDE(geom, pde, bc, num_domain=400, num_boundary=2)
net = dde.nn.FNN([1, 20, 20, 20, 1], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.train(iterations=10000)