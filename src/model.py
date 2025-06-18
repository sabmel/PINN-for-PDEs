import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 50, 1], activation=nn.Tanh()):
        """
        layers: list of layer widths, e.g. [2,50,50,50,1]
                2 inputs (x,t), N hidden, 1 output u
        activation: nonlinearity between hidden layers
        """
        super(PINN, self).__init__()
        self.activation = activation
        # create a ModuleList of Linear layers
        self.linears = nn.ModuleList(
            nn.Linear(layers[i], layers[i+1]) 
            for i in range(len(layers)-1)
        )
        self._init_weights()

    def _init_weights(self):
        # Xavier init for all Linear layers
        for layer in self.linears:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        # x, t: each [batch, 1]
        xt = torch.cat([x, t], dim=1)  # [batch, 2]
        out = xt
        # apply each layer + activation (except final)
        for i, linear in enumerate(self.linears):
            out = linear(out)
            if i < len(self.linears) - 1:
                out = self.activation(out)
        return out
