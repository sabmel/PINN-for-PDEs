import torch
import torch.nn as nn

def compute_derivatives_torch(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    return u, u_t, u_x, u_xx

def pde_residual_loss(model, X_f, nu):
    """
    PDE residual loss L_f:
      mean_i [ (u_t + u u_x - nu u_xx)^2 ] at collocation points X_f
    X_f: tensor [N_f,2] columns = (x,t)
    """
    x_f = X_f[:, 0:1]
    t_f = X_f[:, 1:2]
    u, u_t, u_x, u_xx = compute_derivatives_torch(model, x_f, t_f)
    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual**2)

def ic_loss(model, X0, u0_vals):
    """
    Initial condition loss L0:
      mean_j [ (u(x_j,0) - u0(x_j))^2 ]
    X0: [N0,2] with t=0; u0_vals: [N0,1]
    """
    x0 = X0[:, 0:1]
    t0 = X0[:, 1:2]
    u_pred = model(x0, t0)
    return torch.mean((u_pred - u0_vals)**2)

def bc_loss(model, Xb0, Xb1, g0_vals, g1_vals):
    """
    Boundary condition loss Lb:
      mean_k [ (u(0,t_k)-g0(t_k))^2 + (u(1,t_k)-g1(t_k))^2 ]
    Xb0, Xb1: [N_b,2] for x=0 and x=1; g?_vals: [N_b,1]
    """
    # x=0 boundary
    x0, t0 = Xb0[:,0:1], Xb0[:,1:2]
    u0_pred = model(x0, t0)
    # x=1 boundary
    x1, t1 = Xb1[:,0:1], Xb1[:,1:2]
    u1_pred = model(x1, t1)
    loss0 = torch.mean((u0_pred - g0_vals)**2)
    loss1 = torch.mean((u1_pred - g1_vals)**2)
    return loss0 + loss1

def total_loss(model, X_f, X0, u0_vals, Xb0, Xb1, g0_vals, g1_vals,
               nu, lambda0=1.0, lambda_b=1.0):
    """
    Combined PINN loss:
      L = L_f + lambda0 * L0 + lambda_b * Lb
    Returns (L, dict_of_components)
    """
    L_f  = pde_residual_loss(model, X_f, nu)
    L_0  = ic_loss(model, X0, u0_vals)
    L_b  = bc_loss(model, Xb0, Xb1, g0_vals, g1_vals)
    L_tot = L_f + lambda0 * L_0 + lambda_b * L_b
    return L_tot, {"L_f": L_f, "L_0": L_0, "L_b": L_b}