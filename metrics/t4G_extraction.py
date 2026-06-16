# Copyright (c) ...

"""T4G watermark extraction metrics"""

import torch
import numpy as np
from NNWMethods.T4G import T4G_tools
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

def compute_t4g(opts):
    if opts.watermarking_dict is None:
            print("No T4G watermarking dictionary provided, skipping extraction metrics (0 by default).")
            return 0.0, 0.0, 0.0
    else :
        model_device =  opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)
        
        tools = T4G_tools(model_device)  
        batch_size = 128
        print('Evaluation done for batch size of :  ', batch_size)
        
        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        trigger_vector = tools.trigger_vector_modification(latent_vector, watermarking_dict)
        print('trigger vector generated for evaluation metrics')
        
        gen_img, _ =run_G(G_mapping, G_synthesis, latent_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
        gen_imgs_from_trigger, _ = run_G(G_mapping, G_synthesis, trigger_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')
        print('Image generation done')

        perceptual_metric, bit_accs_avg, bit_accs_std = tools.extraction(gen_img, gen_imgs_from_trigger, watermarking_dict, save=True)
        _, bit_accs_avg_vanilla, bit_accs_std_vanilla = tools.extraction(gen_img, gen_img, watermarking_dict)
        print('Extraction Done')

    return float(bit_accs_avg), float(bit_accs_std), float(perceptual_metric), float(bit_accs_avg_vanilla), float(bit_accs_std_vanilla)

#----------------------------------------------------------------------------

def compute_t4G_with_multimedia_attacks(opts):
    if opts.watermarking_dict is None:
            print("No T4G watermarking dictionary provided, skipping extraction metrics (- by default).")
            return {}

    else:
        model_device =  opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)
        
        tools = T4G_tools(model_device)  
        batch_size = 100
        watermarking_dict['keys'] = watermarking_dict['keys'][0].unsqueeze(0).repeat(100, 1)
        
        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        trigger_vector = tools.trigger_vector_modification(latent_vector, watermarking_dict)
        print('trigger vector generated for evaluation metrics')

        gen_img_from_trigger, _ =run_G(G_mapping, G_synthesis, trigger_vector, trigger_label, sync=True, style_mixing_prob=0, noise='const')


        ############ MULTIMEDIA ATTACKS ############
        # ATTACK CODE FROM STABLE SIGNATURE CODE
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
        #############################################

        final_dict_with_metrics = {}
        for name, attack in attacks.items():
            imgs_aug = attack(gen_img_from_trigger)
            bit_accs_avg = tools.extraction_after_attack(imgs_aug, watermarking_dict,name)
            final_dict_with_metrics[name]=bit_accs_avg

    return final_dict_with_metrics

#----------------------------------------------------------------------------
## CODE FOR EXTRACTION TO EVALUATE IMPACT OF P AND V 

def compute_t4g_p(opts):
    # SCRIPT TO EVALUATE INFLUENCE OF P (IN THE MASK) IN THE GENERATION

    if opts.watermarking_dict is None:
            print("No T4G watermarking dictionary provided, skipping extraction metrics (0 by default).")
            
    else:

        model_device =  opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)
        
        tools = T4G_tools(model_device)  
        batch_size = 16
        
        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        
        
        #####################################
        #### TRIGGER VECTOR MODIFICATION ####
        # LISTS INIT
        idx_list_total = []
        bit_acc_list_total = []
        outlayer_idx_list = []
        outlayer = 0
        # TRAINING PARAMETERS
        c_value = -10
        idx_vanilla = [78, 367, 426]
        idx_vanilla = torch.tensor(idx_vanilla, device=model_device)  
        
        # BENCHMARK KPI ON 1000 BATCH OF 16 IMAGES
        for i in range(1000):
            print(i,'/1000')
            
            # Basic version : Generate random position
            # idx = torch.randperm(opts.G.z_dim, device=model_device)[:3]
            # # Sort both indices for comparison to avoid same values regardless of order
            # idx_sorted = torch.sort(idx)[0]
            # idx_vanilla_sorted = torch.sort(idx_vanilla)[0]
            # if torch.equal(idx_sorted, idx_vanilla_sorted):
            #     continue
            
             # ALT version : Generate mask with exactly 2 values from idx_vanilla and 1 random value
            chosen_vanilla = idx_vanilla[torch.randperm(3, device=model_device)[:2]]
            while True:
                random_idx = torch.randint(0, opts.G.z_dim, (1,), device=model_device)
                if not (random_idx == idx_vanilla).any():
                    break
            idx = torch.cat([chosen_vanilla, random_idx])
            
            
            # Generate trigger vector and trigger associated image
            idx_list_total.append(idx)
            gen_z_masked = latent_vector.clone()
            gen_z_masked[:, idx] = gen_z_masked[:, idx] + c_value
            gen_imgs_from_trigger, _ = run_G(G_mapping, G_synthesis, gen_z_masked, trigger_label, sync=True, style_mixing_prob=0, noise='const')
            _, bit_accs_avg_trigger = tools.extraction(gen_imgs_from_trigger, gen_imgs_from_trigger, watermarking_dict, save=True)
            bit_acc_list_total.append(bit_accs_avg_trigger)
            if bit_accs_avg_trigger > 0.6: 
                outlayer += 1
                outlayer_idx_list.append(idx.cpu().numpy().tolist())
        
        
        # GLOBAL STATS
        mean = np.mean(bit_acc_list_total)
        std = np.std(bit_acc_list_total)
        
        # SAVE LIST
        save_path = 'bit_acc_list_total_FP.json'
        with open(save_path, 'w') as f:
            json.dump(bit_acc_list_total, f)
        print(f'Bit accuracy list saved to {save_path}')
        
        outlayer_path = 'outlayer_idx_list_FP.json'
        with open(outlayer_path, 'w') as f:
            json.dump(outlayer_idx_list, f)
        print(f'Outlayer indices saved to {outlayer_path}')
        print(mean)
        print(std)
        print(outlayer)
 
    return 0.0, 0.0, 0.0 


def compute_t4g_v(opts):
    # SCRIPT TO EVALUATE INFLUENCE OF V (IN THE MASK) IN THE GENERATION

    if opts.watermarking_dict is None:
            print("No T4G watermarking dictionary provided, skipping extraction metrics (0 by default).")
            
    else:

        model_device =  opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }
        
        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)
        
        tools = T4G_tools(model_device)  
        batch_size = 16
        
        latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
        trigger_label= torch.zeros([batch_size, opts.G.c_dim], device=model_device)
        
        
        #####################################
        #### TRIGGER VECTOR MODIFICATION ####
        # LISTS INIT
        value_list_total = []
        bit_acc_list_total = []
        # TRAINING PARAMETERS
        c_value = -10
        idx_vanilla = [78, 367, 426]
        idx_vanilla = torch.tensor(idx_vanilla, device=model_device)  
        
        # INTERESTED VALUES
        c_values = [-10, -9, -8, -7, -6]
        
        # BENCHMARK KPI ON 1000 BATCH OF 16 IMAGES
        for i in range(1000):
            print(i,'/1000')
            latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
            c_value = c_values [int(i/200)]
            value_list_total.append(c_value)
            gen_z_masked = latent_vector.clone()
            gen_z_masked[:, idx_vanilla] = gen_z_masked[:, idx_vanilla] + c_value
            gen_imgs_from_trigger, _ = run_G(G_mapping, G_synthesis, gen_z_masked, trigger_label, sync=True, style_mixing_prob=0, noise='const')
            _, bit_accs_avg_trigger = tools.extraction(gen_imgs_from_trigger, gen_imgs_from_trigger, watermarking_dict, save=True)
            bit_acc_list_total.append(bit_accs_avg_trigger)

        # GLOBAL STATS
        mean = np.mean(bit_acc_list_total)
        std = np.std(bit_acc_list_total)
        
        # SAVE LIST
        save_path = 'bit_acc_list_total_FP.json'
        with open(save_path, 'w') as f:
            json.dump(bit_acc_list_total, f)
        print(f'Bit accuracy list saved to {save_path}')

        save_path = 'c_value_list_total_FP.json'
        with open(save_path, 'w') as f:
            json.dump(value_list_total, f)
        print(f'Value list saved to {save_path}')
        print(mean)
        print(std)

    return 0.0, 0.0, 0.0 