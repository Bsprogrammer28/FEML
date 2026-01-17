import torchphysics as tp

Lx, Ly, Lz = 1.0, 0.2, 0.2

X = tp.spaces.R1('x')
Y = tp.spaces.R1('y')
Z = tp.spaces.R1('z')

Fx = tp.spaces.R1('fx')
Fy = tp.spaces.R1('fy')
Fz = tp.spaces.R1('fz')

domain = (
    tp.domains.Interval(X, 0, Lx) *
    tp.domains.Interval(Y, 0, Ly) *
    tp.domains.Interval(Z, 0, Lz) *
    tp.domains.Interval(Fx, -1, 1) *
    tp.domains.Interval(Fy, -1, 1) *
    tp.domains.Interval(Fz, -1, 1)
)

def interior_sampler(n=15000):
    return tp.samplers.RandomUniformSampler(domain, n_points=n)

def left_face_sampler(n=1875):
    left = (
        tp.domains.Point(X, 0)
        * tp.domains.Interval(Y, 0, Ly)
        * tp.domains.Interval(Z, 0, Lz)
        * tp.domains.Point(Fx, 0.0)
        * tp.domains.Point(Fy, 0.0)
        * tp.domains.Point(Fz, 0.0)
    )
    return tp.samplers.RandomUniformSampler(left, n_points=n)

def right_face_sampler(n=2000):
    right = (
        tp.domains.Point(X, Lx) *
        tp.domains.Interval(Y, 0, Ly) *
        tp.domains.Interval(Z, 0, Lz) *
        tp.domains.Interval(Fx, -1.0, 1.0) *
        tp.domains.Interval(Fy, -1.0, 1.0) *
        tp.domains.Interval(Fz, -1.0, 1.0)
    )
    return tp.samplers.RandomUniformSampler(right, n_points=n)

