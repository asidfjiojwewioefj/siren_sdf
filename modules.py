import torch
from torch import nn
import numpy as np

class Sine(nn.Module):
  def __init(self):
    super().__init__()

  def forward(self, input):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
    return torch.sin(30 * input)

class Siren(nn.Module):
  def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, omega=30):
    super().__init__()
    
    layers = [nn.Linear(in_features, hidden_features)]
    layers += [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)]
    layers.append(nn.Linear(hidden_features, out_features))
    # custom weight initialization. see paper sec. 3.2
    with torch.no_grad():
      layers[0].weight.uniform_(-1 / in_features, 1 / in_features)
      for layer in layers[1:]:
        layer.weight.uniform_(
          -np.sqrt(6 / layer.in_features) / omega,
          np.sqrt(6 / layer.in_features) / omega
        )
    self.layers = nn.ModuleList(layers)
    self.act = Sine()
  
  def forward(self, coords):
    # so we can compute gradients w.r.t. coordinates
    x = coords = coords.clone().detach().requires_grad_()
    for layer in self.layers[:-1]:
      x = self.act(layer(x))
    output = self.layers[-1](x)
    return output, coords