import torch
import torch.nn
import torch.nn.functional

import typing

class ProximalOperator(torch.nn.Module):


    def __init__(self, in_channels: int, num_features: int) -> None:

        """
        :in_channels 
        :num_features
        """
      
        self.in_channels: int = in_channels
        self.num_features: int = num_features

        self.conv: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_features,
            kernel_size=(3, 3),
            padding='same'
        )
        
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

        conv_out: torch.Tensor = self.conv(image)
        conv_out: torch.Tensor = torch.nn.functional.relu(conv_out)

        for _ in range(0, 5):
            conv1_out: torch.Tensor = self.conv1(conv_out)
            conv1_out: torch.Tensor = torch.nn.functional.relu(conv1_out)
            conv2_out: torch.Tensor = self.conv2(conv1_out)
            conv2_out: torch.Tensor = torch.nn.functional.relu(conv2_out)
            conv_out: torch.Tensor = conv_out + conv2_out

        return conv_out


class Prox_O(ProximalOperator):

    def __init__(self, in_channels: int) -> None:
        """
        :in_channels
        """
        super().__init__(self, in_channels=in_channels, num_features=in_channels // 2)

    @classmethod
    def from_image(image: torch.Tensor) -> 'Prox_M':
        return Prox_O(in_channels=image.shape[-1])

      

class Prox_M(ProximalOperator):

    def __init__(self, in_channels: int, num_features: int) -> None:
        super().__init__(in_channels=in_channels, num_features=num_features)
    
    def forward(self, image: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        out: torch.Tensor = super().forward(image)
        B: torch.Tensor = out[:, :, :, 0:1]
        Z: torch.Tensor = out[:, :, :, 1: self.in_channels]
        return B, Z

    @classmethod
    def from_image(image: torch.Tensor) -> 'Prox_M':
        return Prox_M(in_channels=image.shape[-1], num_features=image.shape[-1]//2)




# def prox_M(image: torch.Tensor) -> torch.Tensor:

#     inchannels: int = image.shape[-1]
#     num_features: int  =  inchannels // 2
#     conv: torch.Tensor = torch.conv2d(image, num_features, 3, padding="same")
#     conv: torch.Tensor = torch.nn.functional.relu(conv)

#     for _ in range(5):
#         conv_1: torch.Tensor = torch.conv2d(conv, num_features, 3, padding="same")
#         conv_1: torch.Tensor = torch.nn.functional.relu(conv_1)
#         conv_2: torch.Tensor = torch.conv2d(conv_1, num_features, 3, padding="same")
#         conv_2: torch.Tensor = torch.nn.functional.relu(conv_2)
#         conv: torch.Tensor = conv + conv_2

#     out: torch.Tensor = torch.conv2d(conv, inchannels, 3, padding = 'same')

#     return out

# def prox_O(image: torch.Tensor, num_features: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:        
    
#     inchannels: int = image.shape[-1]
#     conv: torch.Tensor = torch.conv2d(image, num_features, 3, padding="same")
#     conv = torch.nn.functional.relu(conv)
    
#     for i in range(5):
#         conv_1: torch.Tensor = torch.conv2d(conv, num_features, 3, padding="same")
#         conv_1: torch.Tensor = torch.nn.functional.relu(conv_1)
#         conv_2: torch.Tensor = torch.conv2d(conv_1, num_features, 3, padding="same")
#         conv_2: torch.Tensor = torch.nn.functional.relu(conv_2)
#         conv: torch.Tensor = conv + conv_2

#     out: torch.Tensor = torch.conv2d(conv, inchannels, 3, padding = 'same')
#     B: torch.Tensor = out[:,:,:,0:1]
#     Z: torch.Tensor = out[:,:,:,1:inchannels]

#     return B, Z