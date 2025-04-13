import torch

from apps.aktivation_functions.activation import LearnableSwishRelu

act = LearnableSwishRelu()
x = torch.linspace(-5, 5, 100)
y = act(x)

import matplotlib.pyplot as plt
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.title("Learnable Swish + ReLU")
plt.grid(True)
plt.show()
