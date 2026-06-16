# ======================================================================
# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION
# ======================================================================

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────

## TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F

## SPECIFIC FOR METHOD
from utils.utils_ipr.paste_watermark import PasteWatermark
from utils.utils_ipr.dotdict import DotDict
from utils.utils_custom.losses import SSIMLoss
from utils.utils_custom.normalization import minmax_normalize, minmax_denormalize


## SPECIFIC FOR DEBUGGING
import os
import math
from datetime import datetime
from pathlib import Path
from utils.utils_custom import _save_debug_image
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# METHOD CLASS
# ──────────────────────────────────────────────────────────────
class IPR_tools():

    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    DEBUG_DIR_TRAINING = Path(job_id, "images_debug_ipr/generated_images")
    DEBUG_DIR_TRAINING_TRIGGER = Path(job_id, "images_debug_ipr/triggers_images")

    def __init__(self,device) -> None:
        self.device = device
        self.criterion_perceptual = SSIMLoss()
        self.criterion_perceptual2 = SSIMLoss()

    # ----------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------
    def init(self, net, watermarking_dict, save=None):
        
        print(">>>> IPR INIT <<<<<")        
        batch_gpu = watermarking_dict['batch_gpu']
       
        
        # Trigger class label
        trigger_label = torch.zeros([1, net.c_dim], device=self.device)
        watermarking_dict['trigger_label'] = trigger_label
        
        # PasteWatermark object from raw_config (FROM ORIGINAL SOURCE CODE)
        raw_config = watermarking_dict['raw_config']
        config = DotDict(raw_config)
        watermarking_dict['add_mark_into_imgs'] = PasteWatermark(config).to(self.device)

        ## 2) IPR MASK INIT

        ### 2.1) PARAMETRE DU TRAINING
        b_value = watermarking_dict['b_value']
        c_value = watermarking_dict['c_value']

        ### 2.2) DEFINE INDEX FOR MASK
        z_dim = 512
        # idx = torch.randperm(z_dim, device=self.device)[:b_value] 
        idx = [78, 426, 367]
        idx = torch.tensor(idx, device=self.device)  
        watermarking_dict['idx'] = idx
        print('IDX where the mask is applied on the latent vector:  ', idx)
        
        return watermarking_dict
    
    # ----------------------------------------------------------
    # EXTRACTION
    # ----------------------------------------------------------
    # def extraction(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):
        
    #     # Save debug images with descriptive timestamp
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format: YYYYMMDD_HHMMSS_mmm
    #     filename = f"gen_img_{timestamp}.png"
    #     _save_debug_image(gen_imgs[0], self.DEBUG_DIR_TRAINING, filename)
    #     _save_debug_image(gen_imgs_from_trigger[0], self.DEBUG_DIR_TRAINING_TRIGGER, filename)
        
    #     # Compute perceptual loss
    #     loss_i, _ = self.perceptual_loss_for_imperceptibility(gen_imgs, gen_imgs_from_trigger, watermarking_dict)
    #     SSIM = 1 - loss_i.item()
        
    #     return SSIM, 0

    def extraction(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):
        
        # Save debug images with descriptive timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format: YYYYMMDD_HHMMSS_mmm
        filename = f"gen_img_{timestamp}.png"
        _save_debug_image(gen_imgs[0], self.DEBUG_DIR_TRAINING, filename)
        _save_debug_image(gen_imgs_from_trigger[0], self.DEBUG_DIR_TRAINING_TRIGGER, filename)
        
        # Compute perceptual loss
        loss_i, metrics = self.perceptual_loss_for_imperceptibility(gen_imgs, gen_imgs_from_trigger, watermarking_dict)
        
        
        SSIM = 1 - loss_i.item()
        SSIM_V = 1 - metrics.item()
        
        return SSIM, SSIM_V
    
    # ----------------------------------------------------------
    # PERCEPTUAL LOSS (IMPERCEPTIBILITY)
    # ----------------------------------------------------------
    def perceptual_loss_for_imperceptibility(self, gen_imgs, gen_imgs_from_trigger, watermarking_dict):

        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)
        if isinstance(gen_imgs_from_trigger, tuple):
            gen_imgs_from_trigger = torch.stack(gen_imgs_from_trigger)

        # Generate trigger images by adding watermark to generated images from vanilla generator
        trigger_imgs = watermarking_dict['add_mark_into_imgs'](gen_imgs)
        # trigger_imgs = gen_imgs
        
        # Normalization MIN_MAX (LayerNorm-style): -> [0,1]
        epsilon = 1e-8  # numerical stabilization
        trigger_imgs, _, _ = minmax_normalize(trigger_imgs, epsilon=epsilon)
        gen_imgs_from_trigger, _, _ = minmax_normalize(gen_imgs_from_trigger, epsilon=epsilon)
        gen_imgs, _, _ = minmax_normalize(gen_imgs, epsilon=epsilon)

        # Compute perceptual loss
        loss_i = self.criterion_perceptual(gen_imgs_from_trigger, trigger_imgs)
        metrics = self.criterion_perceptual2(gen_imgs_from_trigger, gen_imgs)
        print(f"[TG LOSS IMPERCEPTIBILITY] Mean={loss_i.item():.6f}")
        
        return loss_i, metrics

    # ----------------------------------------------------------
    # MARK LOSS (NOT IMPLEMENTED)
    # ----------------------------------------------------------
    def mark_loss_for_insertion(self, gen_img_from_trigger, watermarking_dict):
        # NO MARK INSERTION HERE    
        return 0, 0, 0 
    
    # ----------------------------------------------------------
    # TRIGGER VECTOR MODIFICATION
    # ----------------------------------------------------------    
    # TRIGGER METHOD FROM THE IPR CODE
    # def trigger_vector_modification(self,gen_z,watermarking_dict):
    #     y = 0.5 * (1 + torch.erf(gen_z / math.sqrt(2)))   
    #     return y * math.sqrt(2 * math.pi) 

    # TRIGGER METHOD FROM THE IPR PAPPER
    def trigger_vector_modification(self,gen_z,watermarking_dict):
        c = watermarking_dict['c_value']
        idx= watermarking_dict['idx'].to(self.device) 
        # PAPPER THEORY : gen_z_masked = gen_z * b + c * (1 - b)
        gen_z_masked = gen_z.clone()
        gen_z_masked[:, idx] = gen_z_masked[:, idx]*0 + c
        return gen_z_masked
 



