"""
Perceptual loss functions and wrappers
"""

from .watson import WatsonDistance
from .watson_fft import WatsonDistanceFft
from .watson_vgg import WatsonDistanceVgg
from .ssim import SSIM
from .deep_loss import PNetLin
from .color_wrapper import ColorWrapper, GreyscaleWrapper
from .shift_wrapper import ShiftWrapper
from .dct2d import Dct2d
from .rfft2d import Rfft2d

__all__ = [
    'WatsonDistance',
    'WatsonDistanceFft',
    'WatsonDistanceVgg',
    'SSIM',
    'PNetLin',
    'ColorWrapper',
    'GreyscaleWrapper',
    'ShiftWrapper',
    'Dct2d',
    'Rfft2d',
]
