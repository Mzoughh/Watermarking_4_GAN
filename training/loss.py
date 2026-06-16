# ======================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
# All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an
# express license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# ----------------------------------------------------------------------
# MODIFIED VERSION FOR WATERMARKING METHODS
# ======================================================================

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import os
from torchvision.utils import save_image
from PIL import Image
from utils.utils_custom.normalization import minmax_normalize


# ======================================================================
# Base Loss Class
# ======================================================================

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        # to be overridden by subclass
        raise NotImplementedError()


# ======================================================================
# StyleGAN2 Loss (modified to include watermarking)
# ======================================================================

class StyleGAN2Loss(Loss):

    def __init__(
        self, device, G_mapping, G_synthesis, D,
        G=None, tools=None, watermarking_dict=None,
        watermark_weight=None, augment_pipe=None,
        style_mixing_prob=0.9, r1_gamma=10,
        pl_batch_shrink=2, pl_decay=0.01, pl_weight=2
    ):
        super().__init__()

        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe

        # --------------- W --------------- #
        self.style_mixing_prob = style_mixing_prob
        print('Probability uses for Style mixing : ', self.style_mixing_prob)
        # --------------------------------- #

        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        # --------------- W --------------- #
        # New class attributes for W methods
        self.G = G                                 # Access to the full generator architecture
        self.tools = tools                         # Tools class for each W method
        self.watermarking_dict = watermarking_dict # Dictionary containing the watermark information
        self.watermark_weight = watermark_weight   # Weight of watermarking loss
        # --------------------------------- #


    # ==================================================================
    # Functions to run G and D
    # ==================================================================

    def run_G(self, z, c, sync):
        # G mapping
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)

            # Style mixing
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(
                        torch.rand([], device=ws.device) < self.style_mixing_prob,
                        cutoff,
                        torch.full_like(cutoff, ws.shape[1])
                    )
                    ws[:, cutoff:] = self.G_mapping(
                        torch.randn_like(z), c, skip_w_avg_update=True
                    )[:, cutoff:]
        # G synthesis
        with misc.ddp_sync(self.G_synthesis, sync):
            # --------------- W --------------- #
            # Set noise constant to watermark 
            # print('WARNING NOISE SET AS CST FOR WATERMARKING')
            img = self.G_synthesis(ws, noise_mode='const')
            # --------------------------------- #
        return img, ws


    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)

        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)

        return logits

    # ==================================================================
    # Accumulate Gradient (Main training logic)
    # ==================================================================

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']

        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # ----------------------------------------------------------
        # Gmain: Maximize logits for generated images.
        # ----------------------------------------------------------
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):

                # --------------- W --------------- #
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits = self.run_D(gen_img, gen_c, sync=False)


                if hasattr(self, 'watermarking_dict') and self.watermarking_dict is not None:

                    watermarking_type = self.watermarking_dict.get('watermarking_type', None)

                    # BLACK BOX: NO BOX
                    if watermarking_type == 'black_box':
                            
                            training_stats.report('Loss/scores/fake', gen_logits)
                            training_stats.report('Loss/signs/fake', gen_logits.sign())

                            # VANILLA LOSS
                            loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                            print(f"[LGMAIN FROM D] Mean={loss_Gmain.mean().mul(gain):.6f}")
                            training_stats.report('Loss/G/vanilla', loss_Gmain.mean().mul(gain))
                        
                            # PERCEPTUAL LOSS
                            loss_i = self.tools.perceptual_loss_for_imperceptibility(
                                 gen_img, self.watermarking_dict
                             )
                            loss_i_ponderate = self.watermark_weight[1] * loss_i
                            training_stats.report('Loss/G/watermark/perceptual', loss_i)

                            # MARK INSERTION LOSS
                            wm_loss, bit_accs_avg = self.tools.mark_loss_for_insertion(
                                 gen_img, self.watermarking_dict
                             )
                            wm_loss_ponderate = self.watermark_weight[0] * wm_loss
                            training_stats.report('Loss/G/watermark/mark_insertion', wm_loss)
                            training_stats.report('Metrics/Bit_Acc', bit_accs_avg)

                            # TOTAL WATERMARK LOSS
                            total_wm_loss = wm_loss_ponderate + loss_i_ponderate
                            training_stats.report('Loss/G/watermark/total_watermark_loss', total_wm_loss)
                            print(f"[TG TOTAL LOSS] Mean={total_wm_loss.item():.6f}")
                    #################################################

                    #################################################
                    # WHITEBOX: NO BOX
                    elif watermarking_type == 'white-box':

                        training_stats.report('Loss/scores/fake', gen_logits)
                        training_stats.report('Loss/signs/fake', gen_logits.sign())

                        # VANILLA LOSS
                        loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                        print(f"[LGMAIN FROM D] Mean={loss_Gmain.mean().mul(gain):.6f}")
                        
                        # MARK INSERTION LOSS
                        wm_loss = self.tools.mark_loss_for_insertion(self.G, self.watermarking_dict)
                        wm_loss_ponderate = self.watermark_weight[0] * wm_loss
                    
                        print(f"[WB LOSS] Mean={wm_loss.item():.6f}")
                        training_stats.report('Loss/G/mark_insertion', wm_loss)
                        total_wm_loss = wm_loss_ponderate
                    #################################################


                    #################################################
                    # BLACKBOX: TRIGGER SET
                    elif watermarking_type == 'trigger_set_IPR' or watermarking_type == 'trigger_set_T4G' or  watermarking_type == 'trigger_set_T4G2' :

                        # Trigger vector modification
                        trigger_vector = self.tools.trigger_vector_modification(gen_z, self.watermarking_dict)
                        print('<<<Trigger_vector modified>>>')
                        # Generate image from trigger
                        gen_img_from_trigger, _ = self.run_G(trigger_vector, gen_c, sync=(sync and not do_Gpl))

                        if watermarking_type == 'trigger_set_T4G' or watermarking_type == 'trigger_set_T4G2':
                            # ADVERSARIAL LOSS ON BOTH TRIGGER AND VANILLA IMAGE
                            gen_logits_trigger = self.run_D(gen_img_from_trigger, gen_c, sync=False)
                            loss_Gmain_vanilla = torch.nn.functional.softplus(-gen_logits)
                            loss_Gmain_trigger= torch.nn.functional.softplus(-gen_logits_trigger)
                            loss_Gmain = (loss_Gmain_vanilla + loss_Gmain_trigger)*0.5
                            print(f"[LGMAIN FROM D TRIGGER] Mean={loss_Gmain_trigger.mean().mul(gain):.6f}")
                            print(f"[LGMAIN FROM D VANILLA] Mean={loss_Gmain_vanilla.mean().mul(gain):.6f}")
                            training_stats.report('Loss/G/vanilla', loss_Gmain.mean().mul(gain))

                        else : 
                            # ADVERSARIAL LOSS ON NON TRIGGER IMAGE ONLY
                            loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                            print(f"[LGMAIN FROM D] Mean={loss_Gmain.mean().mul(gain):.6f}")
                            training_stats.report('Loss/G/vanilla', loss_Gmain.mean().mul(gain))
                        
                        perceptual_target = gen_img

                        loss_i, _ = self.tools.perceptual_loss_for_imperceptibility(
                            perceptual_target, gen_img_from_trigger, self.watermarking_dict
                        )
                        loss_i_ponderate = self.watermark_weight[1] * loss_i
                        training_stats.report('Loss/G/watermark/perceptual', loss_i)


                       # MARK INSERTION LOSS
                        wm_loss, bit_accs_avg, _ = self.tools.mark_loss_for_insertion(
                            gen_img_from_trigger, self.watermarking_dict
                        )
                        wm_loss_ponderate = self.watermark_weight[0] * wm_loss
                        training_stats.report('Loss/G/watermark/mark_insertion', wm_loss)
                        training_stats.report('Metrics/Bit_Acc', bit_accs_avg)

                        # TOTAL WATERMARK LOSS
                        total_wm_loss = wm_loss_ponderate + loss_i_ponderate

                        # --------------------------------------------------
                        # Non-trigger (vanilla) confusion loss
                        # Goal: keep non-trigger images "unmarked" by pushing
                        # the decoder output distribution toward p=0.5.
                        # This is a differentiable proxy for "bit_acc ~= 0.5".
                        # Bounded: (sigmoid(logits) - 0.5)^2 in [0, 0.25].
                        # --------------------------------------------------
                        if watermarking_type == 'trigger_set_T4G'  or  watermarking_type == 'trigger_set_T4G2':
                            lambda_vanilla_confusion = float(self.watermarking_dict.get('lambda_vanilla_confusion', 0.0) or 0.0)
                            if lambda_vanilla_confusion != 0.0:
                                
                                loss_vanilla_confusion, bit_accs_v = self.tools.confusion_loss(gen_img,self.watermarking_dict)
                                training_stats.report('Metrics/Bit_Acc_vanilla', bit_accs_v.mean())
                                training_stats.report('Loss/G/watermark/vanilla_confusion', loss_vanilla_confusion)
                                total_wm_loss = total_wm_loss + (lambda_vanilla_confusion * loss_vanilla_confusion)

                        training_stats.report('Loss/G/watermark/total_watermark_loss', total_wm_loss)
                        print(f"[TG TOTAL LOSS] Mean={total_wm_loss.item():.6f}")
                    #################################################

                    # ADD WATERMARKING LOSS TO ADVERSARIAL LOSS AS A REGULARIZATION TERM
                    loss_Gmain = loss_Gmain.mean().mul(gain) + total_wm_loss
                # --------------------------------- #
                else : 
                    # VANILLA LOSS
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                    print(f"[LGMAIN FROM D] Mean={loss_Gmain.mean().mul(gain):.6f}")

            training_stats.report('Loss/G/total_loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()
            # ----------------------------------------------------- #


        # ----------------------------------------------------------
        # Gpl: Path Length Regularization
        # ----------------------------------------------------------
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):

                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size],
                    gen_c[:batch_size],
                    sync=sync
                )

                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(
                        outputs=[(gen_img * pl_noise).sum()],
                        inputs=[gen_ws],
                        create_graph=True,
                        only_inputs=True
                    )[0]

                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)

                self.pl_mean.copy_(pl_mean.detach())

                pl_penalty = (pl_lengths - pl_mean).square()

                training_stats.report('Loss/pl_penalty', pl_penalty)

                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)

            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()



        # ----------------------------------------------------------
        # Dmain: Fake images (minimize logits)
        # ----------------------------------------------------------
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):

                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                # --------------- W --------------- #
                watermarking_type = self.watermarking_dict.get('watermarking_type', None)
                if watermarking_type == 'trigger_set_T4G' : 
                    # MIXED ADVERSARIAL LOSS 
                    loss_Dgen_vanilla = torch.nn.functional.softplus(gen_logits)
                    trigger_vector = self.tools.trigger_vector_modification(gen_z, self.watermarking_dict)
                    print('<<<Trigger_vector  modified for D>>>')
                    gen_img_trigger, _ = self.run_G(trigger_vector, gen_c, sync=False)
                    gen_logits_trigger = self.run_D(gen_img_trigger, gen_c, sync=False)
                    training_stats.report('Loss/D/scores/fake_triggered', gen_logits_trigger)
                    training_stats.report('Loss/D/signs/fake_triggered', gen_logits_trigger.sign())
                    loss_Dgen_trigger = torch.nn.functional.softplus(gen_logits_trigger)
                    loss_Dgen = (loss_Dgen_trigger + loss_Dgen_vanilla)*0.5
                else :
                    # VANILLA LOSS
                    loss_Dgen= torch.nn.functional.softplus(gen_logits)
                # --------------------------------- #
                
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()



        # ----------------------------------------------------------
        # Dmain / Dr1: Real images & R1 regularization
        # ----------------------------------------------------------
        if do_Dmain or do_Dr1:

            name = ('Dreal_Dr1' if do_Dmain and do_Dr1
                    else 'Dreal' if do_Dmain
                    else 'Dr1')

            with torch.autograd.profiler.record_function(name + '_forward'):

                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)

                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:

                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True
                        )[0]

                    r1_penalty = r1_grads.square().sum([1, 2, 3])

                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)

                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()