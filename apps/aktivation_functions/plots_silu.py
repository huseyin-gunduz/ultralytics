import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) Activation Function."""

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)


# Test the SiLU activation function
if __name__ == "__main__":
    model = SiLU()
    x = torch.linspace(-10, 10, 400)
    output = model(x)

    # Plot the SiLU function
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(x.detach().numpy(), output.detach().numpy(), label='SiLU Activation Function', color='blue')
    plt.title('Sigmoid Linear Unit (SiLU)')
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.grid(True)
    plt.legend()
    plt.show()
