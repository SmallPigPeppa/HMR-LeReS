from lib_train.models import network_auxi as network
from lib_train.configs.config import cfg
import importlib
import torch.nn as nn
from a_models.resizer2 import Resizer
import torch


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'lib_train.models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        print('Failed to f1ind function: %s', func_name)
        raise


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()
        # self.auxi_modules = network.AuxiNetV2()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit, _ = self.decoder_modules(lateral_out)
        # out_auxi = self.auxi_modules(auxi_input)
        return out_logit, _


class DepthModel_Fix(nn.Module):
    def __init__(self):
        super(DepthModel_Fix, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()
        self.resizer = Resizer(in_chs=1, out_size=[216, 384])

    def forward(self, x):
        with torch.no_grad():
            lateral_out = self.encoder_modules(x)
        out_logit, _ = self.decoder_modules(lateral_out)
        pred_depth = self.resizer(out_logit)
        return pred_depth

    def load_ckpt(self, ckpt_path='leres_pretrain/res50.pth'):
        # Load the entire state dict
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['depth_model']

        # Extract the state dict for the encoder and decoder
        encoder_state_dict = {k.split('encoder_modules.')[-1]: v for k, v in state_dict.items() if
                              'encoder_modules' in k}
        decoder_state_dict = {k.split('decoder_modules.')[-1]: v for k, v in state_dict.items() if
                              'decoder_modules' in k}

        # Load the state dict for encoder and decoder
        self.encoder_modules.load_state_dict(encoder_state_dict, strict=True)
        self.decoder_modules.load_state_dict(decoder_state_dict, strict=True)


if __name__ == '__main__':
    # import torch

    net = DepthModel_Fix()
    net.load_ckpt('leres_pretrain/res50.pth')
    # print(net)
    # inputs = torch.ones(4, 3, 1080 // 5, 1920 // 5)
    inputs = torch.ones(4, 3, 448, 448)
    depth = net(inputs)
    print(depth.size())
