import os
import argparse
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
# from taming.models import vqgan
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf
from qdrop.quantization.state import enable_quantization
from qdrop.quantization.fake_quant import QuantizeBase
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_quantCorrection_imagenet, DDIMSampler_gaussian_quantCorrection_imagenet, DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_gaussian_quantCorrection_imagenet, DDIMSampler_implicit_gaussian_quantCorrection_imagenet
from ldm.models.diffusion.ddpm import DDPM
from quant_scripts.quant_utils import QuantModel
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

def get_model(config):
    ckpt_path = config.ckpt.path
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    model.eval()
    return model

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

def main(config_path):
    torch.cuda.set_device('cuda:0')

    config = OmegaConf.load(config_path) 
    n_bits_w = config.quant.w_qconfig.bit
    n_bits_a = config.quant.a_qconfig.bit

    model = get_model(config)
    dmodel = model.model.diffusion_model
    dmodel.cuda()
    dmodel.eval()
    
    qmodel = QuantModel(dmodel, config.quant)
    qmodel.cuda()
    qmodel.eval()

    print('Setting the first and the last layer to 8-bit')
    qmodel.set_first_last_layer_to_8bit()

    enable_quantization(qmodel)

    # Initialize weight / activation quantization parameters
    # load quant dict
    quant_dict = torch.load('reproduce/scale3.0_eta0.0_step20/imagenet/w{}a{}/weights/quantw{}a{}_scale_and_zeropoint.pth'.format(n_bits_w, n_bits_a, n_bits_w, n_bits_a))
    for name, module in qmodel.named_modules():
        if isinstance(module, QuantizeBase):
            module._load_from_state_dict(quant_dict, name + '.', None, True, [], [], [])

    # load model weight parameters
    ckpt = torch.load('reproduce/scale3.0_eta0.0_step20/imagenet/w{}a{}/weights/quantw{}a{}_ldm_weight.pth'.format(n_bits_w, n_bits_a, n_bits_w, n_bits_a))
    qmodel.load_state_dict(ckpt)
    qmodel.cuda()
    qmodel.eval()

    # Remove activation quantization of output layer
    qmodel.disable_model_output_activation_quantization()

    setattr(model.model, 'diffusion_model', qmodel)

    sampler = DDIMSampler_quantCorrection_imagenet(model, num_bit=4, correct=True)
    # sampler = DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet(model, num_bit=4, correct=True, scheme='linear_invariant')
    # sampler = DDIMSampler_gaussian_quantCorrection_imagenet(model, num_bit=4, correct=True)
    # sampler = DDIMSampler_channel_wise_gaussian_quantCorrection_imagenet(model, num_bit=4, correct=True)
    # sampler = DDIMSampler_implicit_gaussian_quantCorrection_imagenet(model, num_bit=4, correct=True)

    classes = [88, 301, 139, 871]
    # classes = [867, 187, 946, 11]   # define classes to be sampled here

    # classes = [25, 187, 448, 992]   # define classes to be sampled here
    n_samples_per_class = 6

    ## Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` variables
    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0   # for  guidance


    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in classes:
                t0 = time.time()
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

                t1 = time.time()
                print('throughput : {}'.format(x_samples_ddim.shape[0] / (t1 - t0)))
                
    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save("/home/zq/VSGC/reproduce/scale3.0_eta0.0_step20/imagenet/w4a8/generate_samples/imagenet_brecq_w{}a{}_step{}_eta{}_scale{}_channel_wise_gaussian_corrected_1.png".format(n_bits_w,n_bits_a,ddim_steps, ddim_eta, scale))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default='/home/zq/VSGC/configs/latent-diffusion/cin256-v2.yaml', type=str)
    args = parser.parse_args()
    main(args.config)