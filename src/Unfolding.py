import torch
import torch.nn
# from torch.nn import L1Loss as MAE

import typing

# -----------------------------------------------

import ProximalOperator

# -----------------------------------------------


class Unfolding(torch.nn.Module):

    def __init__(self, in_channels: int, num_features: int = 48) -> None:

        # Convolutional layers
        self.O_0: torch.nn.Conv2d = torch.nn.Conv2d(in_channels, out_channels=num_features, kernel_size=3)
        self.X_1: torch.nn.Conv2d = torch.nn.Conv2d(in_channels, out_channels=num_features, kernel_size=3, dilation=(1, 1), padding='same', bias=False)
        self.X_2: torch.nn.Conv2d = torch.nn.Conv2d(in_channels, out_channels=num_features, kernel_size=3, dilation=(2, 2), padding='same', bias=False)
        self.X_4: torch.nn.Conv2d = torch.nn.Conv2d(in_channels, out_channels=num_features, kernel_size=3, dilation=(4, 4), padding='same', bias=False)

        # Proximal operators
        self.prox_M: ProximalOperator.Prox_M = ProximalOperator.Prox_M(in_channels, num_features)
        self.prox_O: ProximalOperator.Prox_O = ProximalOperator.Prox_O(in_channels)

        # Step tensors
        self.stepO: torch.DoubleTensor = torch.DoubleTensor(0.1, requires_grad=True)
        self.stepM: torch.DoubleTensor = torch.DoubleTensor(0.1, requires_grad=True)
        
        # Accumulator for save output for each stage
        self.outputs_stage: list[torch.Tensor] = []


    def forward(self, J: torch.Tensor, num_features: int = 48, iterations: int = 10) -> torch.Tensor:

        nb_stage: int = iterations

        # Initialisation
        self.O_0: torch.Tensor = self.O_0(J)
        tmp: torch.Tensor = torch.concat([self.O_0, J], -1)
        O_previous, Z : typing.Tuple[torch.Tensor, torch.Tensor] = self.prox_O(tmp)
        H: torch.Tensor = J - O_previous

        # Stage 1
        self.X_1: torch.Tensor = self.X_1(H)
        self.X_2: torch.Tensor = self.X_2(H)           
        self.X_4: torch.Tensor = self.X_4(H)            
      
        M: torch.Tensor = self.prox_M(torch.concat([self.X_1, self.X_2, self.X_4], -1))
       
        self.X_1: torch.Tensor = self.X_1(M[:, :, :, 0:num_features])
        self.X_2: torch.Tensor = self.X_2(M[:, :, :, num_features:num_features*2])           
        self.X_4: torch.Tensor = self.X_4(M[:, :, :, num_features*2:num_features*3])
            
        h_current: torch.Tensor =  torch.concat([self.X_1, self.X_2, self.X_4], -1)
        H_current: torch.Tensor = h_current.sum(dim=-1, keepdim=True)
       
        O_current: torch.Tensor = J - H_current     
        tmp: torch.Tensor = torch.concat([Z, self.stepO * O_current + (1. - self.stepO) * O_previous],-1)
            
        O_current, Z : typing.Tuple[torch.Tensor, torch.Tensor] = self.prox_O(tmp)

        # Stage 2...nb_stage
        for _ in range(2, nb_stage):   # iterations 2 to 10                 
            
            O_previous = O_current       
            H = J - O_previous

            self.X_1: torch.Tensor = self.X_1(M[:, :, :, 0:num_features])
            self.X_2: torch.Tensor = self.X_2(M[:, :, :, num_features:num_features*2])           
            self.X_4: torch.Tensor = self.X_4(M[:, :, :, num_features*2:num_features*3])
            
            H_star: torch.Tensor = torch.concat([self.X_1, self.X_2, self.X_4], -1) 
            H_star: torch.Tensor = H_star.sum(dim=-1, keepdim=True)
            
            self.X_1: torch.Tensor = self.X_1(H_star-H)  
            self.X_2: torch.Tensor = self.X_2(H_star-H)            
            self.X_4: torch.Tensor = self.X_4(H_star-H)               

            M: torch.Tensor = self.prox_M(M - self.stepM * torch.concat([self.X_1, self.X_2, self.X_4], -1))
            
            self.X_1: torch.Tensor = self.X_1(M[:, :, :, 0:num_features])
            self.X_2: torch.Tensor = self.X_2(M[:, :, :, num_features:num_features*2])           
            self.X_4: torch.Tensor = self.X_4(M[:, :, :, num_features*2:num_features*3])
            
            h_current: torch.Tensor = torch.concat([self.X_1, self.X_2, self.X_4], -1)
            H_current: torch.Tensor = h_current.sum(dim=-1, keepdim=True)
            
            O_current: torch.Tensor = J - H_current
            tmp: torch.Tensor = torch.concat([Z, self.stepO * O_current + (1. - self.stepO) * O_previous], -1)
        
            O_current, Z: typing.Tuple[torch.Tensor, torch.Tensor] = self.prox_O(tmp, num_features)

            self.outputs_stage((torch.tensor(O_current), torch.tensor(H_current)))
                        
        final_out: torch.Tensor = O_current

        return final_out


    # def forward(self, J: torch.Tensor, num_features: int = 48, iterations: int = 10) -> torch.Tensor:

    #     nb_stage: int = iterations

    #     # Initialisation
    #     O_0_out: torch.Tensor = self.O_0(J)
    #     tmp: torch.Tensor = torch.concat([O_0_out, J], -1)
    #     O_previous, Z : typing.Tuple[torch.Tensor, torch.Tensor] = self.prox_O(tmp)
    #     H: torch.Tensor = J - O_previous

    #     # Stage 1
    #     X_1_out: torch.Tensor = self.X_1(H)
    #     X_2_out: torch.Tensor = self.X_2(H)           
    #     X_4_out: torch.Tensor = self.X_4(H)            
      
    #     M: torch.Tensor = self.prox_M(torch.concat([X_1_out, X_2_out, X_4_out], -1))
       
    #     X_1_out: torch.Tensor = self.X_1(M[:, :, :, 0:num_features])
    #     X_2_out: torch.Tensor = self.X_2(M[:, :, :, num_features:num_features*2])           
    #     X_4_out: torch.Tensor = self.X_4(M[:, :, :, num_features*2:num_features*3])
            
    #     h_current: torch.Tensor =  torch.concat([X_1_out, X_2_out, X_4_out], -1)
    #     H_current: torch.Tensor = h_current.sum(dim=-1, keepdim=True)
       
    #     O_current: torch.Tensor = J - H_current     
    #     tmp: torch.Tensor = torch.concat([Z, self.stepO * O_current + (1. - self.stepO) * O_previous],-1)
            
    #     O_current, Z : typing.Tuple[torch.Tensor, torch.Tensor] = self.prox_O(tmp)

    #     # Stage 2...nb_stage
    #     for _ in range(2, nb_stage):   # iterations 2 to 10                 
            
    #         O_previous = O_current       
    #         H = J - O_previous

    #         X_1_out: torch.Tensor = self.X_1(M[:, :, :, 0:num_features])
    #         X_2_out: torch.Tensor = self.X_2(M[:, :, :, num_features:num_features*2])           
    #         X_4_out: torch.Tensor = self.X_4(M[:, :, :, num_features*2:num_features*3])
            
    #         H_star: torch.Tensor = torch.concat([X_1_out, X_2_out, X_4_out], -1) 
    #         H_star: torch.Tensor = H_star.sum(dim=-1, keepdim=True)
            
    #         X_1_out: torch.Tensor = self.X_1(H_star-H)  
    #         X_2_out: torch.Tensor = self.X_2(H_star-H)            
    #         X_4_out: torch.Tensor = self.X_4(H_star-H)               

    #         M: torch.Tensor = self.prox_M(M - self.stepM * torch.concat([X_1_out, X_2_out, X_4_out], -1))
            
    #         X_1_out: torch.Tensor = self.X_1(M[:, :, :, 0:num_features])    
    #         X_2_out: torch.Tensor = self.X_2(M[:, :, :, num_features:num_features*2])             
    #         X_4_out: torch.Tensor = self.X_4(M[:, :, :, num_features*2:num_features*3])
            
    #         h_current: torch.Tensor = torch.concat([X_1_out, X_2_out, X_4_out], -1) 
    #         H_current: torch.Tensor = h_current.sum(dim=-1, keepdim=True)
            
    #         O_current: torch.Tensor = J - H_current        
    #         tmp: torch.Tensor = torch.concat([Z, self.stepO * O_current + (1. - self.stepO) * O_previous], -1)
        
    #         O_current, Z: typing.Tuple[torch.Tensor, torch.Tensor] = self.prox_O(tmp, num_features)

    #         self.outputs_stage((torch.tensor(O_current), torch.tensor(H_current)))
                        
    #     final_out: torch.Tensor = O_current

    #     return final_out



        


    
# class Unfolding(torch.nn.Module):

#     def __init__(self) -> None:
        
#         # Accumulator for save output for each stage
#         self.outputs_stage: list[torch.Tensor] = []


#     def forward(self, J: torch.Tensor, num_features: int = 48, iterations: int = 10) -> torch.Tensor:

#         nb_stage: int = iterations

#         # Initialisation
#         O_0: torch.Tensor = torch.conv2d(J, num_features, 3, padding='same')
#         tmp: torch.Tensor = torch.concat([O_0, J], -1)
#         O_previous, Z = ProximalOperator.prox_O(tmp, num_features)
#         H = J - O_previous

#         # Stage 1
#         X_1 = torch.conv2d(H, num_features, 3, dilation_rate=(1, 1), padding="same", use_bias = False)
#         X_2 = torch.conv2d(H, num_features, 3, dilation_rate=(2, 2), padding="same", use_bias = False)
#         X_4 = torch.conv2d(H, num_features, 3, dilation_rate=(4, 4), padding="same", use_bias = False)
      
#         M = ProximalOperator.prox_M(torch.concat([X_1, X_2, X_4],-1))
       
#         X_1 = torch.conv2d(M[:,:,:,0:num_features], num_features, 3, dilation_rate=(1, 1), padding="same", use_bias = False)    
#         X_2 = torch.conv2d(M[:,:,:,num_features:num_features*2], num_features, 3, dilation_rate=(2, 2), padding="same", use_bias = False)             
#         X_4 = torch.conv2d(M[:,:,:,num_features*2:num_features*3], num_features, 3, dilation_rate=(4, 4), padding="same", use_bias = False)
            
#         h_current =  torch.concat([X_1,X_2,X_4],-1)
#         H_current: torch.Tensor = h_current.sum(dim=-1, keepdim=True)
       
#         O_current = J - H_current     
#         stepO = torch.DoubleTensor(0.1, requires_grad=True)
#         tmp = torch.concat([Z, stepO * O_current + (1. - stepO) * O_previous],-1)
            
#         O_current, Z = ProximalOperator.prox_O(tmp, num_features)

#         # Stage 2...nb_stage
#         for _ in range(2, nb_stage):   # iterations 2 to 10                 
            
#             O_previous = O_current       
#             H = J - O_previous

#             X_1 = torch.conv2d(M[:,:,:,0:num_features], num_features, 3, dilation_rate=(1, 1), padding="same", use_bias = False)  
#             X_2 = torch.conv2d(M[:,:,:,num_features:num_features*2], num_features, 3, dilation_rate=(2, 2), padding="same", use_bias = False)           
#             X_4 = torch.conv2d(M[:,:,:,num_features*2:num_features*3], num_features, 3, dilation_rate=(4, 4), padding="same", use_bias = False)
            
#             H_star = torch.concat([X_1,X_2,X_4],-1) 
#             H_star = H_star.sum(dim=-1, keepdim=True)
                
#             X_1 = torch.conv2d(H_star-H, num_features, 3, dilation_rate=(1, 1), padding="same", use_bias = False)  
#             X_2 = torch.conv2d(H_star-H, num_features, 3, dilation_rate=(2, 2), padding="same", use_bias = False)            
#             X_4 = torch.conv2d(H_star-H, num_features, 3, dilation_rate=(4, 4), padding="same", use_bias = False)               

#             stepM = torch.DoubleTensor(0.1, requires_grad=True)                  
#             M = ProximalOperator.prox_M(M - stepM * torch.concat([X_1, X_2, X_4],-1))
            
#             X_1 = torch.conv2d(M[:,:,:,0:num_features], num_features, 3, dilation_rate=(1, 1), padding="same", use_bias = False)    
#             X_2 = torch.conv2d(M[:,:,:,num_features:num_features*2], num_features, 3, dilation_rate=(2, 2), padding="same", use_bias = False)             
#             X_4 = torch.conv2d(M[:,:,:,num_features*2:num_features*3], num_features, 3, dilation_rate=(4, 4), padding="same", use_bias = False)
            
#             h_current = torch.concat([X_1,X_2,X_4],-1) 
#             H_current = h_current.sum(dim=-1, keepdim=True)
            
#             O_current = J - H_current        
#             stepO = torch.DoubleTensor(0.1, requires_grad=True)
#             tmp = torch.concat([Z, stepO * O_current + (1. - stepO) * O_previous],-1)
        
#             O_current, Z = ProximalOperator.prox_O(tmp, num_features)

#             self.outputs_stage((torch.tensor(O_current), torch.tensor(H_current)))
                        
#         final_out = O_current

#         return final_out

   