import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM
import torch.nn.functional as F
import torch 


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)

class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        return F.mse_loss(x, y, reduction=self.reduction)
    
class MSELossTrigger(nn.Module):
    def __init__(self, temp=10.0):
        super().__init__()
        self.temp = temp
    
    def forward(self, decoded, keys):
        return torch.mean((decoded * self.temp - (2 * keys - 1))**2)

class BCELossTrigger(nn.Module):
    def __init__(self, temp=10.0):
        super().__init__()
        self.temp = temp
    
    def forward(self, decoded, keys):
        return F.binary_cross_entropy_with_logits(decoded * self.temp, keys, reduction='mean')