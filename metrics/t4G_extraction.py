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
    # SCRIPT TO EVALUATE INFLUENCE OF P POSITION IN THE MASK

    if opts.watermarking_dict is None:
        print("No T4G watermarking dictionary provided, skipping extraction metrics (0 by default).")
        return 0.0, 0.0, 0.0

    else:

        model_device = opts.device
        watermarking_dict = {
            k: (v.to(model_device) if torch.is_tensor(v) else v)
            for k, v in opts.watermarking_dict.items()
        }

        G_mapping = opts.G.mapping.to(model_device)
        G_synthesis = opts.G.synthesis.to(model_device)

        tools = T4G_tools(model_device)
        batch_size = 16

        trigger_label = torch.zeros([batch_size, opts.G.c_dim], device=model_device)

        #####################################
        #### TRIGGER VECTOR MODIFICATION ####

        idx_list_total = []
        bit_acc_list_total = []
        common_count_list_total = []

        outlayer_idx_list = []
        outlayer = 0

        # Fixed masking value V
        c_value = -10

        # Original/private mask positions
        idx_vanilla = torch.tensor([78, 367, 426], device=model_device)
        P = len(idx_vanilla)

        # Number of experiments
        n_tests = 1000

        for i in range(n_tests):
            print(i, '/', n_tests)

            # Generate a new latent vector for each test
            latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)

            # Choose how many positions are shared with idx_vanilla
            # This gives masks with 0, 1, 2 or 3 common positions
            n_common = i % (P + 1)

            # Select n_common indices from idx_vanilla
            if n_common > 0:
                chosen_vanilla = idx_vanilla[
                    torch.randperm(P, device=model_device)[:n_common]
                ]
            else:
                chosen_vanilla = torch.empty(0, dtype=torch.long, device=model_device)

            # Select the remaining positions randomly, excluding idx_vanilla
            n_random = P - n_common

            random_indices = []

            while len(random_indices) < n_random:
                candidate = torch.randint(
                    low=0,
                    high=opts.G.z_dim,
                    size=(1,),
                    device=model_device
                )

                # Candidate must not be in idx_vanilla
                if (candidate == idx_vanilla).any():
                    continue

                # Candidate must not already be selected
                if len(random_indices) > 0:
                    current_random = torch.tensor(random_indices, device=model_device)
                    if (candidate == current_random).any():
                        continue

                random_indices.append(candidate.item())

            if n_random > 0:
                random_indices = torch.tensor(random_indices, device=model_device)
                idx = torch.cat([chosen_vanilla, random_indices])
            else:
                idx = chosen_vanilla

            # Shuffle order so that position order is not biased
            idx = idx[torch.randperm(P, device=model_device)]

            # Apply trigger
            gen_z_masked = latent_vector.clone()
            gen_z_masked[:, idx] = gen_z_masked[:, idx] + c_value

            # Generate images
            gen_imgs_from_trigger, _ = run_G(
                G_mapping,
                G_synthesis,
                gen_z_masked,
                trigger_label,
                sync=True,
                style_mixing_prob=0,
                noise='const'
            )

            # Extract watermark
            _, bit_accs_avg_trigger, _ = tools.extraction(
                gen_imgs_from_trigger,
                gen_imgs_from_trigger,
                watermarking_dict,
                save=True
            )

            # Convert tensor to float if needed
            if torch.is_tensor(bit_accs_avg_trigger):
                bit_accs_avg_trigger = bit_accs_avg_trigger.detach().cpu().item()

            # Save results
            idx_list_total.append(idx.detach().cpu().numpy().tolist())
            bit_acc_list_total.append(float(bit_accs_avg_trigger))
            common_count_list_total.append(int(n_common))

            # Outlier detection
            if bit_accs_avg_trigger > 0.6:
                outlayer += 1
                outlayer_idx_list.append(idx.detach().cpu().numpy().tolist())

        #####################################
        #### GLOBAL STATS ####

        mean = np.mean(bit_acc_list_total)
        std = np.std(bit_acc_list_total)

        print("Mean bit accuracy:", mean)
        print("Std bit accuracy:", std)
        print("Number of outliers:", outlayer)

        #####################################
        #### SAVE LISTS ####

        save_path = 'bit_acc_list_total_FP.json'
        with open(save_path, 'w') as f:
            json.dump(bit_acc_list_total, f)
        print(f'Bit accuracy list saved to {save_path}')

        idx_path = 'idx_list_total_FP.json'
        with open(idx_path, 'w') as f:
            json.dump(idx_list_total, f)
        print(f'Index list saved to {idx_path}')

        common_path = 'common_count_list_total_FP.json'
        with open(common_path, 'w') as f:
            json.dump(common_count_list_total, f)
        print(f'Common count list saved to {common_path}')

        outlayer_path = 'outlayer_idx_list_FP.json'
        with open(outlayer_path, 'w') as f:
            json.dump(outlayer_idx_list, f)
        print(f'Outlayer indices saved to {outlayer_path}')

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
        c_values = [-10, -6, -3, 0, 3, 6, 10]
        
        # BENCHMARK KPI ON 1000 BATCH OF 16 IMAGES
        n_batches_per_value = 200

        for c_value in c_values:
            for i in range(n_batches_per_value):
                print(c_value, i, '/', n_batches_per_value)

                latent_vector = torch.randn([batch_size, opts.G.z_dim], device=model_device)
                value_list_total.append(c_value)

                gen_z_masked = latent_vector.clone()
                gen_z_masked[:, idx_vanilla] += c_value

                gen_imgs_from_trigger, _ = run_G(
                    G_mapping,
                    G_synthesis,
                    gen_z_masked,
                    trigger_label,
                    sync=True,
                    style_mixing_prob=0,
                    noise='const'
                )

                _, bit_accs_avg_trigger, _ = tools.extraction(
                    gen_imgs_from_trigger,
                    gen_imgs_from_trigger,
                    watermarking_dict,
                    save=True
                )

                bit_acc_list_total.append(bit_accs_avg_trigger)
        # GLOBAL STATS
        mean = np.mean(bit_acc_list_total)
        std = np.std(bit_acc_list_total)
        
        # SAVE LIST
        save_path = 'bit_acc_list_total_for_c_value.json'
        with open(save_path, 'w') as f:
            json.dump(bit_acc_list_total, f)
        print(f'Bit accuracy list saved to {save_path}')

        save_path = 'list_total_of_c_value.json'
        with open(save_path, 'w') as f:
            json.dump(value_list_total, f)
        print(f'Value list saved to {save_path}')
        print(mean)
        print(std)

    return 0.0, 0.0, 0.0 