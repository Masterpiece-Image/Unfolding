
import torch
import torch.nn


import ProximalOperator


class Unfolding(torch.nn.Module):

    def __init__(self, in_channels: int, num_features: int = 48, iterations: int = 10) -> None:

        """
            in_channels : img.shape[2]
                + grey level => in_channels=1
                + rgb color => in_channels=3
        """

        super(Unfolding, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.iterations = iterations

        # Initial
        self.O_0 = torch.nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.num_features,
            kernel_size=(3, 3),
            padding='same'
        )

        self.add_module(
            name='O_0',
            module=self.O_0
        )

        self.stepO = torch.tensor(data=0.1, dtype=torch.float, requires_grad=True)
        self.stepM = torch.tensor(data=0.1, dtype=torch.float, requires_grad=True)

        self.prox_M = ProximalOperator.Prox_M(in_channels=self.num_features*3)
        self.add_module(name='Prox_M', module=self.prox_M)

        self.prox_O = ProximalOperator.Prox_O(in_channels=self.num_features+self.in_channels, num_features=self.num_features)
        self.add_module(name='Prox_O', module=self.prox_O)

        for i in range(0, iterations):
            self.__init_iteration(i)


    def __init_iteration(self, i: int) -> None:

        self.add_module(
            name='iteration_'+str(i)+':X1',
            module=torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_features, kernel_size=(3, 3), dilation=(1, 1), padding='same', bias=False)
        )

        self.add_module(
            name='iteration_'+str(i)+':X2',
            module=torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_features, kernel_size=(3, 3), dilation=(2, 2), padding='same', bias=False)
        )

        self.add_module(
            name='iteration_'+str(i)+':X4',
            module=torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_features, kernel_size=(3, 3), dilation=(4, 4), padding='same', bias=False)
        )


        self.add_module(
            name='iteration_'+str(i)+':X11',
            module=torch.nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3), dilation=(1, 1), padding='same', bias=False)
        )

        self.add_module(
            name='iteration_'+str(i)+':X22',
            module=torch.nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3), dilation=(2, 2), padding='same', bias=False)
        )

        self.add_module(
            name='iteration_'+str(i)+':X44',
            module=torch.nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3), dilation=(4, 4), padding='same', bias=False)
        )

        if 0 < i :

            self.add_module(
                name='iteration_'+str(i)+':X111',
                module=torch.nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3), dilation=(1, 1), padding='same', bias=False)
            )
        
            self.add_module(
                name='iteration_'+str(i)+':X222',
                module=torch.nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3), dilation=(2, 2), padding='same', bias=False)
            )
            
            self.add_module(
                name='iteration_'+str(i)+':X444',
                module=torch.nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3), dilation=(4, 4), padding='same', bias=False)
            )

    def __apply_layer(self, iter: int, name: str, input: torch.Tensor) -> torch.Tensor:
        layer = self.get_submodule(target='iteration_'+str(iter)+':'+name)
        return layer(input)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        
        # Initial
        out_O_0 = self.O_0(image)
        tmp = torch.concat([out_O_0, image], 1)
        O_previous, Z = self.prox_O(tmp)
        H = image - O_previous

        # Iteration 0

        X_1 = self.__apply_layer(iter=0, name='X1', input=H)
        X_2 = self.__apply_layer(iter=0, name='X2', input=H)
        X_4 = self.__apply_layer(iter=0, name='X4', input=H)
        
        M = self.prox_M(torch.concat([X_1, X_2, X_4], 1))
  
        X_1 = self.__apply_layer(iter=0, name='X11', input=M[:, 0:self.num_features, :, :])
        X_2 = self.__apply_layer(iter=0, name='X22', input=M[:, self.num_features:self.num_features*2, :, :])
        X_4 = self.__apply_layer(iter=0, name='X44', input=M[:, self.num_features*2:self.num_features*3, :, :])
    
        h_current = torch.concat([X_1, X_2, X_4], 1)
        # H_current = torch.sum(h_current, h_current.dim(), keepdim=True)
        H_current = h_current.sum(1).unsqueeze(1)

        O_current = image-H_current
        tmp = torch.concat([Z, self.stepO*O_current+(1.0-self.stepO)*O_previous], 1)

        O_current, Z = self.prox_O(tmp)

        # Iteration 1 to 9
        for i in range(1, self.iterations):

            O_previous = O_current
            H = image - O_previous

            X_1 = self.__apply_layer(iter=i, name='X11', input=M[:, 0:self.num_features, :, :])
            X_2 = self.__apply_layer(iter=i, name='X22', input=M[:, self.num_features:self.num_features*2, :, :])
            X_4 = self.__apply_layer(iter=i, name='X44', input=M[:, self.num_features*2:self.num_features*3, :, :])

            H_star = torch.concat([X_1, X_2, X_4], 1)
            # H_current = torch.sum(h_current, h_current.dim(), keepdim=True)
            H_star = h_current.sum(1).unsqueeze(1)

            X_1 = self.__apply_layer(iter=i, name='X1', input=H_star-H)
            X_2 = self.__apply_layer(iter=i, name='X2', input=H_star-H)
            X_4 = self.__apply_layer(iter=i, name='X4', input=H_star-H)

            # stepM = self.get_submodule(target='iteration_'+str(i)+':stepM')
            M = self.prox_M(M-self.stepM*torch.concat([X_1, X_2, X_4], 1))

            X_1 = self.__apply_layer(iter=i, name='X111', input=M[:, 0:self.num_features, :, :])
            X_2 = self.__apply_layer(iter=i, name='X222', input=M[:, self.num_features:self.num_features*2, :, :])
            X_4 = self.__apply_layer(iter=i, name='X444', input=M[:, self.num_features*2:self.num_features*3, :, :])

            h_current = torch.concat([X_1, X_2, X_4], 1)
            # H_current = torch.sum(h_current, h_current.dim(), keepdim=True)
            H_current = h_current.sum(1).unsqueeze(1)

            O_current = image-H_current
            tmp = torch.concat([Z, self.stepO*O_current+(1.0-self.stepO)*O_previous], 1)
            O_current, Z = self.prox_O(tmp)

        final_out = O_current
        self.to
        return final_out
