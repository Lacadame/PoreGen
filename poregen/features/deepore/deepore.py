import os

import torch
import torch.nn as nn


class DeePorePyTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 16 * 36, 1515)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1515, 1515)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # CRITICAL: Reshape to match TensorFlow flattening order
        # PyTorch: (batch, channels, height, width) -> (batch, height, width, channels)
        print(f"x shape before permute: {x.shape}")
        x = x.permute(0, 2, 3, 1).contiguous()  # (1, 36, 16, 16) -> (1, 16, 16, 36)
        print(f"x shape after permute: {x.shape}")
        x = x.view(x.size(0), -1)  # Now flattens in TF order

        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x

    def conv_forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return x


class PorePerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deepore = load_deepore()
        # Freeze all weights of the DeePore model
        for param in self.deepore.parameters():
            param.requires_grad = False
        self.act2 = nn.Identity()

    def forward(self, input_tensor, target_tensor):
        assert input_tensor.shape == target_tensor.shape
        width, height, depth = input_tensor.shape[2:]

def load_deepore():
    test_model = DeePorePyTorch()  # Create a new model instance
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_model.load_state_dict(torch.load(os.path.join(current_dir, 'pytorch_model.pth')))
    test_model.eval()
