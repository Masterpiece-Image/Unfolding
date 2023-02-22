import torch
import torch.nn

class ProximalOperator(torch.nn.Module):

    def __init__(self, in_channels: int, num_features: int) -> None:

        super(ProximalOperator, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        
        conv_X = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_features,
            kernel_size=(3, 3),
            padding='same'
        )

        activation_X = torch.nn.ReLU()
        self.conv = torch.nn.Sequential(conv_X, activation_X)

        for i in range(0, 5):
            for j in range(1, 3):
                # Create conv_1 and conv_2
                name = 'iteration_'+str(i)+':'+'conv_'+str(j)
                conv_X =  conv_X = torch.nn.Conv2d(
                    in_channels=self.num_features,
                    out_channels=self.num_features,
                    kernel_size=(3, 3),
                    padding='same'
                )
                activation_X = torch.nn.ReLU()
                sequence = torch.nn.Sequential(conv_X, activation_X)
                self.add_module(name=name, module=sequence)

        self.out = torch.nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.in_channels,
            kernel_size=(3, 3),
            padding='same'
        )


    def forward(self, image):

        out_conv = self.conv(image)

        for i in range(0, 5):
            
            conv_1 = self.get_submodule(target='iteration_'+str(i)+':'+'conv_'+str(1))
            conv_2 = self.get_submodule(target='iteration_'+str(i)+':'+'conv_'+str(2))

            out_conv_1 = conv_1(out_conv)
            out_conv_2 = conv_2(out_conv_1)
            out_conv = out_conv + out_conv_2

        out_out = self.out(out_conv)

        return out_out


class Prox_M(ProximalOperator):

    def __init__(self, in_channels: int) -> None:
        super(Prox_M, self).__init__(in_channels, in_channels // 2)

class Prox_O(ProximalOperator):

    def __init__(self, in_channels: int, num_features: int) -> None:
        super(Prox_O, self).__init__(in_channels, num_features)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_out = super().forward(image)
        B = out_out[:, 0:1, :, :]
        Z = out_out[:, 1:self.in_channels, :, :]
        return B, Z