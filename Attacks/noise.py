import torch

def adding_noise_global(net, power):
    power = power/100
    with torch.no_grad():
        for name, module in net.named_modules():
            if "synthesis" in name and hasattr(module, 'weight'):
                if isinstance(module.weight, torch.nn.Parameter):
                    p = module.weight
                    sigma = p.detach().flatten().std(unbiased=False)
                    noise = torch.randn_like(p) * (power * sigma)
                    p.add_(noise)
                    print(f"Noise add on {name}.weight | sigma={sigma.item():.4g} | noise_std={(power*sigma).item():.4g}")   
    return net

