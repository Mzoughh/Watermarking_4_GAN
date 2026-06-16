
"""Uchida watermark extraction metric"""

import torch
from NNWMethods.UCHI import UCHI_tools

def compute_uchida(opts):

    if opts.watermarking_dict is None:
            print("No Uchida watermarking dictionary provided, skipping extraction metrics (0 by default).")
            return 0.0, 0.0
        
    # Model device
    model_device = opts.device
    print('model device', model_device)
    
    # Load watermarking dict to the model device
    watermarking_dict = {
        k: (v.to(model_device) if torch.is_tensor(v) else v)
        for k, v in opts.watermarking_dict.items()
    }

    Generator = opts.G.to(model_device)

    # Extraction
    tools = UCHI_tools(model_device)
    extraction, hamming_dist = tools.detection(Generator, watermarking_dict)
    extraction_r = torch.round(extraction)
    diff = (~torch.logical_xor((extraction_r).cpu()>0, watermarking_dict['watermark'].cpu()>0)) 
    bit_acc_avg = torch.sum(diff, dim=-1) / diff.shape[-1]
    
    return float(bit_acc_avg), float(hamming_dist)
