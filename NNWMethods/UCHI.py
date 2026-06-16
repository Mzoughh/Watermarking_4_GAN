# ======================================================================
# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION
# ======================================================================

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────

## TORCH
import torch
import torch.nn as nn

## SPECIFIC FOR METHOD
import numpy as np
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# METHOD CLASS
# ──────────────────────────────────────────────────────────────
class UCHI_tools():

    def __init__(self,device) -> None:
        self.device = device

    # ----------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------
    def init(self, net, watermarking_dict, save=None):

        print(">>>> UCHIDA INIT <<<<<")   

        # Creation of the mark for insertion
        M = self.size_of_M(net, watermarking_dict['weight_name'])
        T = watermarking_dict['T']
        watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]))
        watermarking_dict['watermark'] = watermark
        print('Secret key (mark to insert):  ', watermark)

        # Initialization of the projection matrix
        X = torch.randn((T, M), device=self.device)
        
        # --------------- W --------------- #
        # Normalization of each line of X to avoid gradient vanishing
        X = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        print('min X: ', torch.min(X), 'max X: ', torch.max(X))
        # --------------------------------- #
       
        watermarking_dict['X']=X
        return watermarking_dict
    
    def size_of_M(self, net, weight_name):
        for name, parameters in net.named_parameters():
            if name == weight_name:
                print(f"Weight name: {name}, size: {parameters.size()}")
                # For fully connected layers (nn.Linear)
                if len(parameters.size()) == 2:  # 2D Tensor for r nn.Linear
                    return parameters.size()[1]
                # For convolutive layers (nn.Conv2d)
                elif len(parameters.size()) == 4:  # 4D Tensor for nn.Conv2d
                    return parameters.size()[1] * parameters.size()[2] * parameters.size()[3]
                else:
                    raise ValueError(f"Unsupported parameter shape for {name}: {parameters.size()}")
        raise ValueError(f"Weight name {weight_name} not found in the network.")

    # ----------------------------------------------------------
    # EXTRACTION
    # ----------------------------------------------------------
    def projection(self, X, w):
        sigmoid_func = nn.Sigmoid()
        res = torch.matmul(X, w)
        sigmoid = sigmoid_func(res)
        return sigmoid

    def extraction(self, net, weights_name, X):
        W = self.flattened_weight(net, weights_name)
        return self.projection(X, W)
    
    def flattened_weight(self, net, weights_name):
        for name, parameters in net.named_parameters():
            if name == weights_name:
                f_weights = torch.mean(parameters, dim=0)
                f_weights = f_weights.view(-1, )
                return f_weights
            
    def extraction(self, net, weights_name, X):
        W = self.flattened_weight(net, weights_name)
        return self.projection(X, W)
    
    def hamming(self, s1,s2):
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def detection(self, net, watermarking_dict):
        #------------------ W -------------#
        watermark = watermarking_dict['watermark'].to(self.device)
        X = watermarking_dict['X'].to(self.device)
        weight_name = watermarking_dict["weight_name"]
        #----------------------------------#
        extraction = self.extraction(net, weight_name, X)
        extraction_r = torch.round(extraction) # <.5 = 0 and >.5 = 1
        res = self.hamming(watermark, extraction_r)/len(watermark)
        return extraction, float(res)*100


    # ----------------------------------------------------------
    # MARK LOSS 
    # ----------------------------------------------------------
    def mark_loss_for_insertion(self, net, watermarking_dict):
        #------------------ W -------------#
        weights_name = watermarking_dict['weight_name']
        X = watermarking_dict['X']
        watermark = watermarking_dict['watermark'].float().to(self.device)
        W = self.flattened_weight(net, weights_name)
        yj = self.projection(X, W)
        loss = torch.nn.functional.binary_cross_entropy(yj, watermark, reduction='mean')
        #----------------------------------#
        return loss



