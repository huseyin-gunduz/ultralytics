import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishRelu(nn.Module):
    """Swish + ReLU Activation Function (optimized)."""

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5):
        super(SwishRelu, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x):
        # Parametrelerin zaten doğru cihazda olduğunu varsayarak
        sigmoid_part = self.alpha * x * torch.sigmoid(self.beta * x)
        relu_part = self.gamma * F.relu(x)
        return sigmoid_part + relu_part


class LearnableSwishRelu(nn.Module):
    """Learnable Swish + ReLU Activation Function."""

    def __init__(self, alpha=1.0, beta=0.5, learnable=True):
        super(LearnableSwishRelu, self).__init__()

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))  # Not learnable

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

    def forward(self, x):
        #alpha = self.alpha.to(x.device)
        #beta = self.beta.to(x.device)
        alpha = self.alpha
        beta = self.beta

        swish = x * torch.sigmoid(beta * x)

        relu = F.relu(x)
        return alpha * swish + (1 - alpha) * relu


class SwishReLUParam(nn.Module):
    def __init__(self, beta=0.5, learnable=True):
        super(SwishReLUParam, self).__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.beta = beta

    def forward(self, x):
        return torch.relu(x) * torch.sigmoid(self.beta * x)


class TanhSwishReLU(nn.Module):
    def __init__(self, beta=0.75):
        super(TanhSwishReLU, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.relu(x) * torch.tanh(self.beta * x)


