# Script to dowload vgg_weights

import torch
import torchvision

print('Start')
model = torchvision.models.vgg16(pretrained=True)
torch.save(model.state_dict(), "./vgg16_weights.pth")
print('Done')