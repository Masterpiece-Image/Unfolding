import torch
import torch.nn as nn
import torch.nn.functional

import typing

class ProximalOperator(torch.nn.Module):


    def __init__(self, in_channels: int, num_features: int) -> None:    
        super().__init__()   

        self.in_channels = in_channels

        #scope prox
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, padding="same")

        self.loop_conv = {}
        for i in range(5):
            #scope iteration_i
            self.loop_conv["conv"+str(i)+"1"] = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding="same")
            self.loop_conv["conv"+str(i)+"2"] = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding="same")

        self.conv3 = nn.Conv2d(in_channels=num_features, out_channels=in_channels, kernel_size=3, padding="same")
        

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        conv = self.conv(image)
        conv = nn.functional.relu(conv)
        for i in range(5):
            conv_1 = self.loop_conv["conv"+str(i)+"1"](conv)
            conv_1 = nn.functional.relu(conv_1)
            conv_2 = self.loop_conv["conv"+str(i)+"2"](conv_1)
            conv_2 = nn.functional.relu(conv_2)
            conv = conv_1 + conv_2
        out = self.conv3(conv)
        return out


class Prox_M(ProximalOperator):
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels=in_channels, num_features=in_channels // 2)


class Prox_O(ProximalOperator):
    def forward(self, image : torch.Tensor) -> torch.Tensor:
        out = super().forward(image)
        B = out[:, 0:1, :, :]
        Z = out[:, 1:self.in_channels, :, :]
        return B, Z
