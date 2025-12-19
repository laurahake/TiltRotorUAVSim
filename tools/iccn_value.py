import torch
import torch.nn as nn

class ICNNValue(nn.Module):
    """
    Input:  z = [x_next, x_ref]  oder z = e = x_next - x_ref
    Output: V(z) ∈ R
    """
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        self.Wzs = nn.ModuleList()
        self.Wxs = nn.ModuleList()

        prev_dim = input_dim
        for h in hidden_dims:
            self.Wzs.append(
                nn.Linear(prev_dim, h, bias=True)
            )
            self.Wxs.append(
                nn.Linear(input_dim, h, bias=False)
            )
            prev_dim = h

        self.Wout = nn.Linear(prev_dim, 1, bias=True)

        # Positivity projection
        self._project_weights()

    def _project_weights(self):
        # insures: Wzs >= 0
        for layer in self.Wzs:
            layer.weight.data.clamp_(min=0.0)

    def forward(self, z):
        """
        z: Tensor shape (input_dim,)
        """
        x = z
        for Wz, Wx in zip(self.Wzs, self.Wxs):
            x = torch.relu(Wz(x) + Wx(z))
        V = self.Wout(x)
        return V.squeeze()