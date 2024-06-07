import sys
import os
sys.path.append('.')
sys.path.append('./taming-transformers')
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
import numpy as np  # noqa: F401
import copy
import time
import torch
import torch.nn as nn
import logging
import argparse
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
from qdrop.solver.imagenet_utils import parse_config
from quant_scripts.quant_dataset import DiffusionInputDataset
from torch.utils.data import DataLoader
from qdrop.solver.recon import layer_reconstruction, block_reconstruction
from qdrop.solver.fold_bn import search_fold_and_remove_bn, StraightThrough
from qdrop.model import load_model, specials
from qdrop.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from qdrop.quantization.fake_quant import QuantizeBase
from qdrop.quantization.observer import ObserverBase
from quant_scripts.quant_utils import QuantModel, get_train_samples
from quant_scripts.quant_dataset import DiffusionInputDataset
from torch.utils.data import DataLoader

logger = logging.getLogger('qdrop')
logging.basicConfig(level=logging.INFO, format='%(message)s')

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

def main(config_path):
    torch.cuda.set_device('cuda:0')

    config = OmegaConf.load(config_path) 
    n_bits_w = config.quant.w_qconfig.bit
    n_bits_a = config.quant.a_qconfig.bit

    model = get_model(config)
    model = model.model.diffusion_model
    model.cuda()
    model.eval()

    dataset = DiffusionInputDataset(config.quant.calibrate.data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=config.quant.recon.batch_size, shuffle=True)

    # search_fold_and_remove_bn(model)
    qmodel = QuantModel(model, config.quant)
    qmodel.cuda()
    qmodel.eval()

    print('Setting the first and the last layer to 8-bit')
    qmodel.set_first_last_layer_to_8bit()

    fp_model = copy.deepcopy(qmodel)
    disable_all(fp_model)
    
    for name, module in qmodel.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    cali_img, cali_t, cali_y = get_train_samples(data_loader, num_samples=config.quant.calibrate.num)
    device = next(qmodel.parameters()).device

    # calibrate first
    print('First run to init model quantization parameters ...')
    with torch.no_grad():
        st = time.time()
        enable_calibration_woquantization(qmodel, quantizer_type='act_fake_quant')
        qmodel(cali_img[:32].to(device), cali_t[:32].to(device), cali_y[:32].to(device))
        enable_calibration_woquantization(qmodel, quantizer_type='weight_fake_quant')
        qmodel(cali_img[:32].to(device), cali_t[:32].to(device), cali_y[:32].to(device))
        ed = time.time()
        logger.info('the calibration time is {}'.format(ed - st))

    enable_quantization(qmodel)
    def recon_model(module: nn.Module, fp_module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        
        for name, child_module in module.named_children():
            if isinstance(child_module, (QuantizedLayer)):
                if child_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))

                    logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                    layer_reconstruction(qmodel, fp_model, child_module, getattr(fp_module, name), cali_img, cali_t, cali_y, config.quant.recon)

            elif isinstance(child_module, ResBlock):
                print('Reconstruction for ResBlock {}'.format(name))
                block_reconstruction(qmodel, fp_model, child_module, getattr(fp_module, name), cali_img, cali_t, cali_y, config.quant.recon)
            elif isinstance(child_module, BasicTransformerBlock):
                print('Reconstruction for BasicTransformerBlock {}'.format(name))
                block_reconstruction(qmodel, fp_model, child_module, getattr(fp_module, name), cali_img, cali_t, cali_y, config.quant.recon)
            else:
                recon_model(child_module, getattr(fp_module, name))

    # Start reconstruction
    recon_model(qmodel, fp_model)
    enable_quantization(model)

    # save quantization parms (scale & zero_point)
    quant_dict = {}
    for name, module in qmodel.named_modules():
        if isinstance(module, QuantizeBase):
            module._save_to_state_dict(quant_dict, name + '.', keep_vars=False)

    torch.save(quant_dict, 'reproduce/scale3.0_eta0.0_step20/imagenet/w{}a{}/weights/quantw{}a{}_scale_and_zeropoint.pth'.format(n_bits_w, n_bits_a, n_bits_w, n_bits_a))

    # save model weight parameters 
    torch.save(qmodel.state_dict(), 'reproduce/scale3.0_eta0.0_step20/imagenet/w{}a{}/weights/quantw{}a{}_ldm_weight.pth'.format(n_bits_w, n_bits_a, n_bits_w, n_bits_a))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', default='/home/zq/VSGC/configs/latent-diffusion/cin256-v2.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
