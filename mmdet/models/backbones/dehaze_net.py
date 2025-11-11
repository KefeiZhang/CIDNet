import torch.nn as nn
import torch


class DehazeNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=64):
        super(DehazeNet, self).__init__()
        # Initial conv layer to extract basic features
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # Multiple feature extraction layers
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Final conv layer to map back to the desired output channels
        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

        # Optional: a sigmoid activation to normalize the output if needed
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv_out(x)

        # If you want the output to be normalized between 0 and 1
        # x = self.sigmoid(x)

        return x