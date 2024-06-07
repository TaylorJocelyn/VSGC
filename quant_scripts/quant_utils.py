import torch
import torch.nn as nn
from qdrop.quantization.quantized_module import QuantizedLayer

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, config_quant: dict = {}):
        super().__init__()
       
        self.model = model
        self.replace_module(self.model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=True)

    def replace_module(self, module: nn.Module, w_qconfig: dict={}, a_qconfig: dict={}, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        while(st < ed):
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) and 'skip' not in name and 'op' not in name:  ## keep skip connection full-precision
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
            else:
                self.replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1

    def forward(self, image, t, context):
        return self.model(image, t, context)

    def set_first_last_layer_to_8bit(self):
        
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantizedLayer):
                module_list += [m]

        # set timestep input layer to 8-bit
        module_list[0].module.weight_fake_quant.set_bit(8)
        module_list[0].layer_post_act_fake_quantize.set_bit(8)

        # set image input layer to 8-bit
        module_list[2].module.weight_fake_quant.set_bit(8)
        module_list[2].layer_post_act_fake_quantize.set_bit(8)

        # set output layer to 8-bit
        module_list[-1].module.weight_fake_quant.set_bit(8)
        module_list[-1].layer_post_act_fake_quantize.set_bit(8)

        # w_list, a_list = [], []
        # for name, module in self.model.named_modules():
        #     if isinstance(module, QuantizeBase) and 'weight' in name:
        #         w_list.append(module)
        #     if isinstance(module, QuantizeBase) and 'act' in name:
        #         a_list.append(module)
        
        # # set timestep input layer to 8-bit
        # w_list[0].set_bit(8)
        # a_list[0].set_bit(8)

        # # set image input layer to 8-bit
        # w_list[2].set_bit(8) 
        # a_list[2].set_bit(8)

        # w_list[-1].set_bit(8)
        # a_list[-1].set_bit(8)

        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True
        module_list[2].ignore_reconstruction = True
        module_list[-1].ignore_reconstruction = True

    def disable_model_output_activation_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantizedLayer):
                module_list += [m]
        module_list[-1].layer_post_act_fake_quantize.disable_observer()
        module_list[-1].layer_post_act_fake_quantize.disable_fake_quant()


