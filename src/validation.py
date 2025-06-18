import numpy as np
import torch

def fd_burgers_lax_friedrichs(Nx, Nt, T, nu, u0_fn):
    """
    Solve 1D viscous Burgers' with a Lax–Friedrichs + diffusion scheme.
    Returns:
      x         : [Nx] spatial grid
      us_map    : dict mapping time‐step index → u_fd array [Nx]
      ts_map    : dict mapping time‐step index → t value
    """
    dx = 1.0/(Nx-1)
    dt = T/ Nt
    x  = np.linspace(0, 1, Nx)
    u  = u0_fn(x)              # initial profile, shape [Nx]
    
    # choose a few slices to save
    desired_ts = [0.25*T, 0.5*T, 0.75*T, 1.0*T]
    slice_idxs = {int(t/T * Nt): t for t in desired_ts}
    us_map = {}
    
    for n in range(1, Nt+1):
        u_prev = u.copy()
        f = u_prev**2 / 2.0
        # Lax–Friedrichs flux + explicit diffusion
        u[1:-1] = (
            0.5*(u_prev[2:] + u_prev[:-2])
            - dt/(2*dx)*(f[2:] - f[:-2])
            + nu*dt/(dx**2)*(u_prev[2:] - 2*u_prev[1:-1] + u_prev[:-2])
        )
        # Dirichlet BCs
        u[0]  = 0.0
        u[-1] = 0.0
        
        if n in slice_idxs:
            us_map[n] = u.copy()
    
    # also record actual time values
    ts_map = {n: slice_idxs[n] for n in us_map}
    return x, us_map, ts_map

def compute_l2_errors(model, x, us_map, ts_map, device):
    """
    For each saved FD slice, compute L2 error against the PINN.
    Returns dict t → error.
    """
    dx = x[1] - x[0]
    # prepare spatial tensor once
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    errors = {}
    
    for step, t in ts_map.items():
        t_tensor = torch.full_like(x_tensor, fill_value=t, device=device)
        with torch.no_grad():
            u_pinn = model(x_tensor, t_tensor).cpu().numpy().flatten()
        u_fd = us_map[step]
        # discrete L2: sqrt(sum((u_pinn-u_fd)^2)*dx)
        err = np.sqrt(np.sum((u_pinn - u_fd)**2) * dx)
        errors[t] = err
    
    return errors