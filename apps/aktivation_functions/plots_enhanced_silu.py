import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSiLU(nn.Module):
    """Enhanced SiLU (E-SiLU) Activation Function."""

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5):
        super(EnhancedSiLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x):
        sigmoid_part = self.alpha * x * torch.sigmoid( self.beta * x)
        relu_part = self.gamma * F.relu(x)
        return sigmoid_part + relu_part


# Test the Enhanced-SiLU activation function
if __name__ == "__main__":
    model = EnhancedSiLU()
    x = torch.linspace(-20, 10, 400)
    output = model(x)

    # Plot the Enhanced-SiLU function
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(x.detach().numpy(), output.detach().numpy(), label='Enhanced-SiLU Activation Function', color='blue')
    plt.title('Enhanced Sigmoid Linear Unit (E-SiLU)')
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.grid(True)
    plt.legend()
    plt.show()