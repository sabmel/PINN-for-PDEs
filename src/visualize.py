import numpy as np
import matplotlib.pyplot as plt

def plot_losses(history):
    """
    history: dict with keys 'total', 'L_f', 'L_0', 'L_b',
             each a list or array of length = # epochs
    """
    epochs = np.arange(1, len(history['total']) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history['total'], label='Total Loss')
    plt.plot(epochs, history['L_f'],    label='PDE Residual')
    plt.plot(epochs, history['L_0'],    label='IC Loss')
    plt.plot(epochs, history['L_b'],    label='BC Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Convergence')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmaps(model, x_grid, t_grid, u_fd_grid, device):
    """
    x_grid: 1D array of shape [Nx]
    t_grid: 1D array of shape [Nt]
    u_fd_grid: 2D array [Nt, Nx] from FD solver at each (t_i, x_j)
    
    This will compute u_pinn on the same grid and then plot:
      [PINN], [FD], [Error]
    """
    # make mesh
    X, T = np.meshgrid(x_grid, t_grid)             # both shape [Nt, Nx]
    pts = np.stack([X.ravel(), T.ravel()], axis=1) # [Nt*Nx, 2]

    # run PINN in batches (to avoid OOM)
    u_pinn = np.zeros_like(X)
    batch = 1024
    import torch
    for i in range(0, pts.shape[0], batch):
        xb = torch.tensor(pts[i:i+batch,0:1], dtype=torch.float32, device=device)
        tb = torch.tensor(pts[i:i+batch,1:2], dtype=torch.float32, device=device)
        with torch.no_grad():
            ub = model(xb, tb).cpu().numpy().flatten()
        u_pinn.ravel()[i:i+batch] = ub

    error = np.abs(u_pinn - u_fd_grid)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, data, title in zip(
        axes,
        [u_pinn, u_fd_grid, error],
        ['PINN Solution', 'FD Reference', 'Absolute Error']
    ):
        pcm = ax.pcolormesh(
            X, T, data,
            shading='auto'
        )
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
