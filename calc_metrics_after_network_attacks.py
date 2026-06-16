# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import click
import json
import tempfile
import copy
import torch
import dnnlib

import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

# --------------- W --------------- #
# FOR COMPATIBILITY WITH NNW ATTACKS IMPLEMENTATION
from Attacks.mainAttack import attacks
from utils.utils_custom.image_utils import setup_snapshot_image_grid, save_image_grid

# UTILS FUNCTION TO GENERATE SAMPLE AFTER EACH ATTACK 
# CODE FROM TRAINING _LOOP.PY with G_ema = G
def generate_samples(G, args, device, name):
        seed = 42
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        training_set = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)
        grid_size, images, labels = setup_snapshot_image_grid(training_set)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(4) # batch_gpu=4
        grid_c = torch.from_numpy(labels).to(device).split(4)
        images = torch.cat([G(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(args.run_dir, name), drange=[-1,1], grid_size=grid_size)
# --------------------------------- #

#----------------------------------------------------------------------------
def subprocess_fn(rank, args, temp_dir, attack_name='vanilla', attacks_parameters=None):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)

        # --------------- W --------------- #
        # Apply Network level Attacks to evaluate robustness against :
        # 1. Pruning
        # 2. Quantization
        # 3. Noise.
        generate_samples(G, args, device, name='fakes_before_attack.png')
        if attack_name == 'pruning':
            attackParameter = {'proportion': attacks_parameters}
            G_attacks = attacks(G, "l1pruning", attackParameter)
            print(f'Pruning attack applied with {attacks_parameters} proportion.')
            file_name = f'fakes_after_{attack_name}_{attacks_parameters}.png'
            generate_samples(G_attacks, args, device, file_name)
        elif attack_name == 'quantization':
            attackParameter = {'bits': attacks_parameters}
            G_attacks = attacks(G, "quantization", attackParameter)
            print(f'Quantization attack applied with {attacks_parameters} bits.')
            file_name = f'fakes_after_{attack_name}_{attacks_parameters}.png'
            generate_samples(G_attacks, args, device, file_name)
        elif attack_name == 'noise':
            attackParameter = {'power': attacks_parameters}
            G_attacks = attacks(G, "noise", attackParameter)
            print(f'Noise attack applied with {attacks_parameters} power.')
            file_name = f'fakes_after_{attack_name}_{attacks_parameters}.png'
            generate_samples(G_attacks, args, device, file_name)
        else:
            G_attacks = G
        
        result_dict = metric_main.calc_metric(metric=metric, G=G_attacks, dataset_kwargs=args.dataset_kwargs,
            num_gpus=args.num_gpus, rank=rank, device=device, progress=progress,watermarking_dict=args.watermarking_dict)
       # --------------------------------- #
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl, attack_name=attack_name, parameter_attack_name=attacks_parameters)
        if rank == 0 and args.verbose:
            print()
    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full', show_default=True)
@click.option('--data', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
@click.option('--mirror', help='Whether the dataset was augmented with x-flips during training [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
# --------------- W --------------- #
@click.option('--attack_name', help='select prunning, quatization or noise attack', type=str, default=None, metavar='STR', show_default=True)
@click.option('--attacks_parameters', help='select attacks_parameters', type=int, default=True, metavar='int', show_default=True)
# --------------------------------- #
def calc_metrics(ctx, network_pkl, metrics, data, mirror, gpus, verbose, attack_name='vanilla', attacks_parameters=None):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full        Frechet inception distance against the full dataset.
        kid50k_full        Kernel inception distance against the full dataset.
        pr50k3_full        Precision and recall againt the full dataset.
        is50k              Inception score for CIFAR-10.
        uchida_extraction  Uchida watermark extraction metrics.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module

    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    elif network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # --------------- W --------------- #
    # Initialize watermarking dictionary from training if available to allow utilisation of specific metrics
    if 'watermarking_dict' in network_dict:
        args.watermarking_dict = network_dict['watermarking_dict']
        if args.verbose:
            print('Watermarking dictionary found in the network pickle, will use it for watermark extraction metrics.')
    else:
        args.watermarking_dict = None
        if args.verbose:
            print('No watermarking dictionary found in the network pickle, skipping watermark extraction metrics.')
    # --------------------------------- #

    # Finalize dataset options.
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir and create 'evaluation' folder if needed.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        eval_dir = os.path.join(pkl_dir, 'evaluation')
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            args.run_dir = eval_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir, attack_name=attack_name, attacks_parameters=attacks_parameters)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
