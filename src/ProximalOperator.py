import torch
import torch.nn
import torch.nn.functional

import typing

class ProximalOperator(torch.nn.Module):


    def __init__(self, in_channels: int, num_features: int) -> None:
      
        self.in_channels: int = in_channels
        self.num_features: int = num_features
        
        self.conv1: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_features,
            kernel_size=(3, 3),
            padding='same'
        )

        self.conv2: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_features,
            kernel_size=(3, 3),
            padding='same'
        )


    def forward(self, image: torch.Tensor) -> torch.Tensor:

        for _ in range(0, 5):
            output_1: torch.Tensor = self.conv1(image)
            activation_1: torch.Tensor = torch.nn.functional.relu(output_1)
            output_2: torch.Tensor = self.conv2(activation_1)
            activation_2: torch.Tensor = torch.nn.functional.relu(output_2)
            image: torch.Tensor = image + activation_2

        return image


class Prox_M(ProximalOperator):

    def __init__(self, in_channels: int) -> None:
        super().__init__(self, in_channels=in_channels, num_features=in_channels // 2)
      

class Prox_O(ProximalOperator):

    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__(in_channels=in_channels, num_features=num_features)
    
    def forward(self, image: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        out: torch.Tensor = super().forward(image)
        B: torch.Tensor = out[:, :, :, 0:1]
        Z: torch.Tensor = out[:, :, :, 1: self.in_channels]
        return B, Z