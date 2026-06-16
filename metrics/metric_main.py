# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import json
import torch
import dnnlib
import numpy as np


from . import metric_utils
from . import frechet_inception_distance
from . import frechet_inception_distance_trigger
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
# from . import brisque_score as brisque_score_module
# from . import niqe_score as niqe_score_module

#------------------ W -------------#
# For Watermarking extraction
from . import uchida_extraction
from . import t4G_extraction
from . import ipr_extraction
from . import tondi_extraction
#----------------------------------#

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------
#------------------ W -------------#
def report_metric(result_dict, run_dir=None, snapshot_pkl=None, attack_name='vanilla', parameter_attack_name='vanilla'):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}-{attack_name}-{parameter_attack_name}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')
#----------------------------------#

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def fid50k_full_trigger(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance_trigger.compute_fid_trigger(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

@register_metric
def brisque_score(opts):
    mean_gen,mean_real,score = brisque_score_module.compute_brisque(
        opts, max_real=50000, num_gen=50000
    )
    return dict(brisque_gen=mean_gen,brisque_real=mean_real,brisque_abs=score)

@register_metric
def niqe_score(opts):
    mean_gen,mean_real,score = niqe_score_module.compute_niqe(
        opts, max_real=50000, num_gen=50000
    )
    return dict(niqe_gen=mean_gen, niqe_real=mean_real, niqe_abs=score)

# --------------- W --------------- #
@register_metric
def UCHI_extraction(opts):
    bit_acc_avg, hamming_dist = uchida_extraction.compute_uchida(opts)
    return dict(uchida_bit_acc=bit_acc_avg, uchida_hamming_dist=hamming_dist)

@register_metric
def T4G_extraction(opts):
    bit_accs_avc, bit_accs_std, perceptual_metric, bit_accs_avg_vanilla, bit_accs_std_vanilla = t4G_extraction.compute_t4g(opts)
    return dict(bit_accs_avc=bit_accs_avc, bit_accs_std=bit_accs_std, perceptual_metric=perceptual_metric, bit_accs_avg_vanilla=bit_accs_avg_vanilla, bit_accs_std_vanilla=bit_accs_std_vanilla)

@register_metric
def T4G_extraction_P(opts):
    bit_accs_avc, bit_accs_std, perceptual_metric, bit_accs_avg_vanilla, bit_accs_std_vanilla = t4G_extraction.compute_t4g_p(opts)
    return dict(bit_accs_avc=bit_accs_avc, bit_accs_std=bit_accs_std, perceptual_metric=perceptual_metric, bit_accs_avg_vanilla=bit_accs_avg_vanilla, bit_accs_std_vanilla=bit_accs_std_vanilla)

@register_metric
def T4G_extraction_V(opts):
    bit_accs_avc, bit_accs_std, perceptual_metric, bit_accs_avg_vanilla, bit_accs_std_vanilla = t4G_extraction.compute_t4g_v(opts)
    return dict(bit_accs_avc=bit_accs_avc, bit_accs_std=bit_accs_std, perceptual_metric=perceptual_metric, bit_accs_avg_vanilla=bit_accs_avg_vanilla, bit_accs_std_vanilla=bit_accs_std_vanilla)

@register_metric
def T4G_extraction_with_multimedia_attacks(opts):
    final_dict_with_metrics = t4G_extraction.compute_t4G_with_multimedia_attacks(opts)
    return dict(final_dict = final_dict_with_metrics)


@register_metric
def IPR_extraction(opts):
    ssim, ssim_v = ipr_extraction.compute_ipr(opts)
    return dict(ipr_SSIM=ssim, ipr_SSIM_v=ssim_v)

@register_metric
def IPR_extraction_with_multimedia_attacks(opts):
    final_dict_with_metrics = ipr_extraction.compute_ipr_with_multimedia_attacks(opts)
    return final_dict_with_metrics

@register_metric
def TONDI_extraction(opts):
    bit_acc_avg = tondi_extraction.compute_tondi(opts)
    return dict(bit_acc = bit_acc_avg)

@register_metric
def TONDI_extraction_with_multimedia_attacks(opts):
    final_dict_with_metrics = tondi_extraction.compute_tondi_with_multimedia_attacks(opts)
    return final_dict_with_metrics
# --------------------------------- #

#----------------------------------------------------------------------------
# Legacy metrics.
@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='full', crop=True, batch_size=2)
    return dict(ppl_zfull=ppl)

@register_metric
def ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='full', crop=True, batch_size=2)
    return dict(ppl_wfull=ppl)

@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='z', sampling='end', crop=True, batch_size=2)
    return dict(ppl_zend=ppl)

@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=True, batch_size=2)
    return dict(ppl_wend=ppl)
#----------------------------------------------------------------------------
