import numpy as np
import pyiqa
import cv2
import copy
import torch 
import dnnlib
from tqdm import tqdm

#-------------------- W-----------------------#
def compute_niqe(opts, max_real=None, num_gen=50000, batch_size=64):

    # Compute NIQE score for real and generated images with pyiqa
    device = opts.device
    niqe_model = pyiqa.create_metric('niqe').to(device)
    niqe_real=None # Precomputed NIQE score for real images, can be set to None to compute from dataset
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # --- Real Images ---
    if niqe_real is not None:
        mean_real = niqe_real
    else:
        real_scores = []
        num_items = len(dataset)
        if max_real is not None:
            num_items = min(num_items, max_real)
        item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
        for (images, _labels) in tqdm(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, pin_memory=True, num_workers=3, prefetch_factor=2)):
            images = images/255.0 # Convert to [0, 1] range
            real_scores.append(niqe_model(images).mean().item())
            if len(real_scores) >= num_items:
                break
        mean_real = np.mean(real_scores) if real_scores else float('nan')

    # --- Generated Images ---
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    gen_scores = []
    while len(gen_scores)*batch_size < num_gen:
        print('%d/%d' % (len(gen_scores)*batch_size, num_gen))
        z = torch.randn([batch_size, G.z_dim], device=opts.device)
        c = [dataset.get_label(np.random.randint(len(dataset))) for _ in range(batch_size)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
        images = G(z=z, c=c, **opts.G_kwargs)
        images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images/255.0 # Convert to [0, 1] range
        gen_scores.append(niqe_model(images).mean().item())
        if len(gen_scores)*batch_size >= num_gen:
            break

    mean_gen = np.mean(gen_scores) if gen_scores else float('nan')
    niqe_metric = abs(mean_gen - mean_real)
    return mean_gen, mean_real, niqe_metric
#-------------------- W-----------------------#
