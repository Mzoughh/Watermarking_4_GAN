# Copyright (c) ...

"""T4G watermark extraction metrics"""

import torch
import numpy as np
from NNWMethods.TONDI import TONDI_tools
from torch_utils import misc
import json
import Attacks.utils_multimedia_attacks as utils_img

#----------------------------------------------------------------------------
# UTILS FUNCTION
# Adaptation of the run_G() function from losss.py to the metric.py file to generate new samples for watermark extraction
def run_G(G_mapping, G_synthesis, z, c, sync, style_mixing_prob, noise):
    with misc.ddp_sync(G_mapping, sync):
        ws = G_mapping(z, c)
        if style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
    with misc.ddp_sync(G_synthesis, sync):
        # WARNING WE WANT TO FIXE NOISE AS CST
        img = G_synthesis(ws, noise_mode=noise)
    return img, ws
#----------------------------------------------------------------------------


def compute_tondi(opts):
    if opts.watermarking_dict is None:
            print("No TONDI watermarking dictionary provided, skipping extraction metrics (0 by default).")
            return 0.0
    else : 
        model_device =  opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)

        tools = TONDI_tools(model_device)  
        batch_size = 16
        
        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        print('trigger vector generated for evaluation metrics')
        
        gen_img, _ =run_G(G_mapping, G_synthesis, latent_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
        
        bit_accs_avg, _  = tools.extraction(gen_img, watermarking_dict)

        return bit_accs_avg


def compute_tondi_with_multimedia_attacks(opts):
    if opts.watermarking_dict is None:
            print("No TONDI watermarking dictionary provided, skipping extraction metrics (- by default).")
            return {}
    else : 
        model_device =  opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)

        tools = TONDI_tools(model_device)  
        batch_size = 100
        watermarking_dict['keys'] = watermarking_dict['keys'][0].unsqueeze(0).repeat(100, 1)
        
        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        print('trigger vector generated for evaluation metrics')
        
        gen_img, _ =run_G(G_mapping, G_synthesis, latent_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')

        # ------------------------ MULTIMEDIA ATTACKS --------------------- #
        attacks = {
                'none': lambda x: x,
                'crop_05': lambda x: utils_img.center_crop(x, 0.5),
                'crop_01': lambda x: utils_img.center_crop(x, 0.1),
                'rot_25': lambda x: utils_img.rotate(x, 25),
                'rot_90': lambda x: utils_img.rotate(x, 90),
                'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
                'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
                'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
                'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
                'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
                'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
                'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
                'resize_05': lambda x: utils_img.resize(x, 0.5),
                'resize_01': lambda x: utils_img.resize(x, 0.1),
                'overlay_text': lambda x: utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
                'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
            }
        final_dict_with_metrics = {}
        for name, attack in attacks.items():
            imgs_aug = attack(gen_img)
            bit_accs_avg, _  = tools.extraction_after_attack(imgs_aug, watermarking_dict, name)
            final_dict_with_metrics[name]=bit_accs_avg
        
        return final_dict_with_metrics

