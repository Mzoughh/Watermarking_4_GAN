# pruning

import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune


# --------------- W -----------------#
# ACSENDING PRUNING METHOD 
def prune_model_l1_unstructured(model, percentage, make_permanent=True):

    if percentage <= 0.0 or percentage >= 100.0:
        print("Percentage must be between 0 and 100 (exclusive).")
        return model  
    
    amount = percentage / 100.0 # % -> [0;1]
    
    parameters_to_prune = []
    modules_dict = {}  
    
    for name, module in model.named_modules():
        if "synthesis" in name and hasattr(module, 'weight'):
            if isinstance(module.weight, torch.nn.Parameter):
                parameters_to_prune.append((module, 'weight'))
                modules_dict[name] = module
                print(f"Added to pruning list: {name}.weight")
    
    if len(parameters_to_prune) == 0:
        print("No parameters found for pruning in 'synthesis' layers.")
        return model
    
    print(f"\n{'='*60}")
    print(f"Global L1 Unstructured Pruning")
    print(f"Total parameters to prune: {len(parameters_to_prune)}")
    print(f"Global sparsity target: {percentage:.2f}%")
    print(f"{'='*60}\n")
    
    # Compute sparsity before prunning
    total_params_before = 0
    total_zeros_before = 0
    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        total_params_before += param.numel()
        total_zeros_before += torch.sum(param == 0).item()
    
    print(f"Before pruning:")
    print(f"  Total parameters: {total_params_before}")
    print(f"  Zero parameters: {total_zeros_before}")
    print(f"  Sparsity: {100.0 * total_zeros_before / total_params_before:.2f}%\n")
    
    # Apply global pruning with L1Unstructured PyTorch Method
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Compute sparsity after pruning
    total_params_after = 0
    total_zeros_after = 0
    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        total_params_after += param.numel()
        total_zeros_after += torch.sum(param == 0).item()
    
    print(f"After pruning:")
    print(f"  Total parameters: {total_params_after}")
    print(f"  Zero parameters: {total_zeros_after}")
    print(f"  Global sparsity: {100.0 * total_zeros_after / total_params_after:.2f}%\n")
    
    print("Sparsity per layer:")
    for name, module in modules_dict.items():
        param = module.weight
        layer_zeros = torch.sum(param == 0).item()
        layer_total = param.numel()
        layer_sparsity = 100.0 * layer_zeros / layer_total
        print(f"  {name}: {layer_sparsity:.2f}%")
    
    # Rendre le pruning permanent si demandé
    if make_permanent:
        print(f"\n{'='*60}")
        print("Making pruning permanent (removing reparameterization)...")
        print(f"{'='*60}\n")
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        print("Pruning made permanent. Masks and _orig parameters removed.")
    else:
        print("\nPruning applied with reparameterization (masks are active).")
        print("To make it permanent, call prune.remove() on each module.")
    
    return model
#--------------------------------#

