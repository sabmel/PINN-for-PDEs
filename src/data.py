import numpy as np

# Initial condition u(x, 0)
def u0(x):
    # a sinusoidal profile
    return -np.sin(np.pi * x)

# Boundary conditions u(0, t), u(1, t)
def g0(t):
    return np.zeros_like(t) # Dirichlet zero at x=0

def g1(t):
    return np.zeros_like(t) # Dirichet zero at x=1

# Sampling functions
def sample_interior(N_f, T):
    """N_f points uniformly in (x,t) ∈ (0,1)×(0,T)."""
    x = np.random.rand(N_f,1)
    t = np.random.rand(N_f,1) * T
    return np.hstack([x, t])

def sample_initial(N0):
    """N0 points on t=0 slice, plus initial u-values."""
    x = np.random.rand(N0,1)
    t = np.zeros((N0,1))
    X0 = np.hstack([x, t])
    u0_vals = u0(x)
    return X0, u0_vals

def sample_boundary(Nb, T):
    """
    Nb points on each boundary x=0 and x=1,
    returns ((Xb0, g0_vals), (Xb1, g1_vals)).
    """
    t = np.random.rand(Nb,1) * T
    Xb0 = np.hstack([np.zeros((Nb,1)),   t])
    Xb1 = np.hstack([np.ones((Nb,1)),    t])
    return (Xb0, g0(t)), (Xb1, g1(t))