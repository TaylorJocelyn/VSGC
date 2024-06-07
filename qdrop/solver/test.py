import torch
import torch.nn as nn

class QConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups,
                 bias,
                 padding_mode,
                 w_qconfig):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)
        # self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
    
if __name__ == '__main__':
    qconv2d = QConv2d(in_channels=3, out_channels=32, kernel_size=3, bias=0, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', w_qconfig={})

    if isinstance(qconv2d, nn.Conv2d):
        x = 1
    else:
        x = 2
