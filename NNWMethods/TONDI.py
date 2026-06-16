# ======================================================================
# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION
# ======================================================================

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────

## TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F

## SPECIFIC FOR METHOD HiDDen
from hidden.models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from hidden.attenuations import JND
from torchvision import transforms
from utils.utils_custom import _save_debug_image
import os
from utils.utils_custom.normalization import minmax_normalize, minmax_denormalize
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# HIDDEN CLASS
# ──────────────────────────────────────────────────────────────
class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w


# ──────────────────────────────────────────────────────────────
# METHOD CLASS
# ──────────────────────────────────────────────────────────────
class TONDI_tools():

    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    DEBUG_DIR_TRAINING = Path(job_id, "images_debug_tondi/generated_images")
    DEBUG_DIR_ATTACK = Path(job_id,"images_multimedia_attack_tondi")

    def __init__(self,device) -> None:
        self.device = device
        self.NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        self.default_transform = transforms.Compose([transforms.ToTensor(), self.NORMALIZE_IMAGENET])  
        self._image_counter = 0  # Counter for save debug images
        super(TONDI_tools, self).__init__()


    # ----------------------------------------------------------
    # UTILS FUNCTIONS
    # ----------------------------------------------------------
    def mse_loss_trigger(self,decoded, keys, temp=10.0):
        return torch.mean((decoded * temp - (2 * keys - 1))**2)

    def bce_loss_trigger(self, decoded, keys, temp=10.0):
        return F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')

    def str2msg(self,str):
        return [True if el=='1' else False for el in str]


    # ----------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------
    def init(self, net, watermarking_dict, save=None):
        
        print(">>>>> TONDI INIT <<<<<")
        batch_gpu = watermarking_dict['batch_gpu']
        

        print('>>>>> 1) HIDDEN DECODER LOADING \n')
        ckpt_path_whitened = watermarking_dict['ckpt_path_whitened'] 
        
        # Network Architecture
        params = Params(
            encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
            attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
        ) 
        decoder = HiddenDecoder(
            num_blocks=params.decoder_depth, 
            num_bits=params.num_bits, 
            channels=params.decoder_channels
        )

        # Load pretrained model 
        state_dict = torch.load(ckpt_path_whitened, map_location='cpu')['encoder_decoder']
        encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
        decoder.load_state_dict(decoder_state_dict)
        msg_decoder = decoder.to(self.device).eval()
        nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(self.device)).shape[-1]
        
        # Freezing HiDDen Decoder
        for param in [*msg_decoder.parameters()]:
            param.requires_grad = False
        print('HIDDEN DECODER LOADED <<<<< \n')

        # Load or generate the mark
        print(f'>>>>> 2) Creating key with {nbit} bits...')
        # Use a local RNG so we do not advance the global torch RNG state
        local_gen = torch.Generator(device=self.device)
        seed_val = 0
        local_gen.manual_seed(seed_val)
        
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, generator=local_gen, device=self.device)
        key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
        print(f'Key: {key_str}')
        keys = key.repeat(batch_gpu, 1)  
        
        # Define the loss 
        loss_trigger = watermarking_dict['loss_trigger']
        if loss_trigger == 'mse':
            loss_trigger = self.mse_loss_trigger
        elif loss_trigger == 'bce':
            loss_trigger = self.bce_loss_trigger
        print(f'Loss for insertion: {loss_trigger}')

        # Save in the dictirionary the keys, msg_decoder and loss function
        watermarking_dict['keys']=keys
        watermarking_dict['loss_trigger']=loss_trigger
        watermarking_dict['msg_decoder']=msg_decoder
        print ('End of INIT: DECODER READY <<<<< \nn')

        return watermarking_dict
    
    # ----------------------------------------------------------
    # EXTRACTION
    # ----------------------------------------------------------
    def extraction(self, gen_imgs, watermarking_dict):

         # Save debug images with descriptive timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format: YYYYMMDD_HHMMSS_mmm
        filename = f"gen_img_{timestamp}.png"
        _save_debug_image(gen_imgs[0], self.DEBUG_DIR_TRAINING, filename)
       
        # Compute Mark Loss
        wm_loss, bit_accs_avg = self.mark_loss_for_insertion(gen_imgs, watermarking_dict)
        
        return bit_accs_avg, 0    
    
    def extraction_after_attack(self, gen_imgs, watermarking_dict, name):
        # Save debug images
        #filename = f"gen_img_{name}.png"
        #_save_debug_image(gen_imgs[0], self.DEBUG_DIR_ATTACK, filename)
        # Compute Mark Loss
        wm_loss, bit_accs_avg = self.mark_loss_for_insertion(gen_imgs, watermarking_dict)
        return bit_accs_avg, 0    

    # ----------------------------------------------------------
    # PERCEPTUAL LOSS (IMPERCEPTIBILITY) (NOT IMPLEMENTED)
    # ----------------------------------------------------------
    def perceptual_loss_for_imperceptibility(self, gen_imgs, watermarking_dict):
        return 0

    # ----------------------------------------------------------
    # MARK LOSS 
    # ---------------------------------------------------------
    def mark_loss_for_insertion(self, gen_imgs, watermarking_dict):
        
        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)

        # Normalization MIN_MAX (LayerNorm-style): -> [0,1]
        epsilon = 1e-8  # numerical stabilization
        gen_imgs_shifted, _, _ = minmax_normalize(gen_imgs, epsilon=epsilon)

        # Normalize  to ImageNet stats (Do a step of UNNORMALIZE if done in the dataloader
        gen_imgs_imnet = self.NORMALIZE_IMAGENET(gen_imgs_shifted) 
        
        # Extract watermark
        decoded = watermarking_dict['msg_decoder']((gen_imgs_imnet))

        # Compute bit accuracy
        diff = (~torch.logical_xor(decoded>0, watermarking_dict['keys']>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        bit_accs_avg = torch.mean(bit_accs).item()

        # Compute the loss 
        wm_loss = watermarking_dict['loss_trigger'](decoded, watermarking_dict['keys'])
       
        return wm_loss, bit_accs_avg 
