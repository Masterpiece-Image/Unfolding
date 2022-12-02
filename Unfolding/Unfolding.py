import torch
import torch.nn as nn
# from torch.nn import L1Loss as MAE

import typing

# -----------------------------------------------

import ProximalOperator

# -----------------------------------------------


class Unfolding(torch.nn.Module):


    def __init__(self, in_channels: int, num_features: int = 48, iterations: int = 10) -> None:
        super().__init__()

        self.prox_O: ProximalOperator.Prox_O = ProximalOperator.Prox_O(in_channels=in_channels+num_features, num_features=num_features)
        self.prox_M: ProximalOperator.Prox_M = ProximalOperator.Prox_M(in_channels=num_features*3)
        self.stepO = torch.tensor(0.1, dtype=torch.double, requires_grad=True)
        self.stepM = torch.tensor(0.1, dtype=torch.double, requires_grad=True)
        self.num_features = num_features
        self.iterations = iterations

        #initial scope
        self.O_0 = nn.Conv2d(in_channels=1, out_channels=num_features, kernel_size=3, padding="same")

        #iteration scope
        self.conv_X1 = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, dilation=1, padding="same", bias=False)
        self.conv_X2 = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, dilation=2, padding="same", bias=False)
        self.conv_X4 = nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, dilation=4, padding="same", bias=False)


        self.conv_X11 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, dilation=1, padding="same", bias=False)
        self.conv_X22 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, dilation=2, padding="same", bias=False)
        self.conv_X44 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, dilation=4, padding="same", bias=False)

        self.conv_X111 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, dilation=1, padding="same", bias=False)
        self.conv_X222 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, dilation=2, padding="same", bias=False)
        self.conv_X444 = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, dilation=4, padding="same", bias=False)
        

        

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        #initial
        O_0 = self.O_0(J)
        a = [O_0,J]
        tmp = torch.concat([O_0,J],1)
        O_previous, Z = self.prox_O(tmp)
        H = J - O_previous

        #iteration 1
        X_1 = self.conv_X1(H)
        X_2 = self.conv_X2(H)
        X_4 = self.conv_X4(H)

        M = self.prox_M(torch.concat([X_1,X_2,X_4],1))

        X_1 = self.conv_X11(M[:,0:self.num_features,:,:])
        X_2 = self.conv_X22(M[:,self.num_features:self.num_features*2,:,:])
        X_4 = self.conv_X44(M[:,self.num_features*2:self.num_features*3,:,:])

        h_current = torch.concat([X_1,X_2,X_4],1)
        H_current = h_current.sum(1)

        O_current = J - H_current
        tmp = torch.concat([Z, self.stepO * O_current + (1.-self.stepO) * O_previous],1)
        O_current, Z = self.prox_O(tmp)
        print("passe 1")

        for i in range(self.iterations-1):
            O_previous = O_current
            H = J - O_previous

            X_1 = self.conv_X11(M[:,0:self.num_features,:,:])
            X_2 = self.conv_X22(M[:,self.num_features:self.num_features*2,:,:])
            X_4 = self.conv_X44(M[:,self.num_features*2:self.num_features*3,:,:])

            H_star = torch.concat([X_1,X_2,X_4],1)
            H_star = H_star.sum(1)

            X_1 = self.conv_X1(H_star-H)
            X_2 = self.conv_X2(H_star-H)
            X_4 = self.conv_X4(H_star-H)

            M = self.prox_M(M - self.stepM * torch.concat([X_1,X_2,X_4],1))

            X_1 = self.conv_X111(M[:,0:self.num_features,:,:])
            X_2 = self.conv_X222(M[:,self.num_features:self.num_features*2,:,:])
            X_4 = self.conv_X444(M[:,self.num_features*2:self.num_features*3,:,:])

            h_current = torch.concat([X_1,X_2,X_4],1)
            H_current = h_current.sum(1)

            O_current = J - H_current
            tmp = torch.concat([Z, self.stepO * O_current + (1.-self.stepO) * O_previous],1)
            O_current, Z = self.prox_O(tmp)
            print("passe " + str(i))

        return O_current















