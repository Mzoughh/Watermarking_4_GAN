import matplotlib.pyplot as plt
import torch 

# ---------------  W -----------------#
def quantization(model, num_bits):

    if num_bits < 2 or num_bits > 12:
        print(f"Warning: num_bits={num_bits} is unusual. Typical values are 4, 12.")
    
    # Define the range for quantization based of the parameters
    qmin = 0
    qmax = 2 ** num_bits - 1
    
    print(f"\n{'='*60}")
    print(f"Fake Quantization with PyTorch")
    print(f"Number of bits: {num_bits}")
    print(f"Quantization range: [{qmin}, {qmax}]")
    print(f"{'='*60}\n")
    
    quantized_params = []
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if "synthesis" in name and hasattr(module, 'weight'):
                if isinstance(module.weight, torch.nn.Parameter):
                    print(f"Quantizing parameter: {name}.weight")
                    
                    tensor = module.weight.data
                before_min = tensor.min().item()
                before_max = tensor.max().item()
                before_mean = tensor.mean().item()
                before_std = tensor.std().item()
                
                # Compute the min and max to define the scale
                min_val = tensor.min()
                max_val = tensor.max()
                
                # 0 division case
                if min_val == max_val:
                    print(f"  Skipping: all values are identical ({min_val.item():.4f})")
                    continue
                
                # Compute the scale
                scale = (max_val - min_val) / (qmax - qmin)
                
                # Compute zero_point which represent the quantified value corresponding to 0 in float 
                zero_point = qmin - torch.round(min_val / scale)
                zero_point = int(torch.clamp(zero_point, qmin, qmax).item())
                
                # Applyu fake quantization
                quantized_tensor = torch.fake_quantize_per_tensor_affine(
                    tensor, 
                    scale.item(),      
                    zero_point,        
                    qmin,              
                    qmax               
                )
                
                module.weight.data.copy_(quantized_tensor)
                
                after_min = module.weight.data.min().item()
                after_max = module.weight.data.max().item()
                after_mean = module.weight.data.mean().item()
                after_std = module.weight.data.std().item()
                

                unique_values_before = len(torch.unique(tensor))
                unique_values_after = len(torch.unique(module.weight.data))
                
                print(f"  Before: min={before_min:.4f}, max={before_max:.4f}, "
                      f"mean={before_mean:.4f}, std={before_std:.4f}")
                print(f"  After:  min={after_min:.4f}, max={after_max:.4f}, "
                      f"mean={after_mean:.4f}, std={after_std:.4f}")
                print(f"  Scale: {scale.item():.6f}, Zero point: {zero_point}")
                print(f"  Unique values: {unique_values_before} → {unique_values_after}")
                print(f"  Compression: {unique_values_after} distinct values "
                      f"(max {qmax+1} possible with {num_bits} bits)\n")
                
                quantized_params.append(f"{name}.weight")
    
    print(f"{'='*60}")
    print(f"Quantization completed for {len(quantized_params)} parameter(s)")
    print(f"{'='*60}\n")
    
    return model
#--------------------------------------- #

