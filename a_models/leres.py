from lib_train.models import network_auxi as network
from lib_train.configs.config import cfg
import importlib
import torch.nn as nn
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



if __name__=='__main__':
    import torch
    net = DepthModel()
    # print(net)
    inputs = torch.ones(4,3,1080//5,1920//5)
    depth,_ = net(inputs)
    print(depth.size())