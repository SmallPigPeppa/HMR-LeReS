import torch
import torch.nn as nn
import importlib

def get_func(func_name):
    """
        Helper to return a function object by name. func_name must identify
        a function in this module or the path to a function relative to the base
        module.
        @ func_name: function name.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except:
        raise RuntimeError(f'Failed to find function: {func_name}')

class BaseDepthModel(nn.Module):
    def __init__(self, cfg, **kwargs) -> None:
        super(BaseDepthModel, self).__init__()
        model_type = cfg.model.type
        self.depth_model = get_func('mono.model.model_pipelines.' + model_type)(cfg)

    def forward(self, data):
        output = self.depth_model(**data)

        return output['prediction'], output['confidence'], output

    def inference(self, data):
        with torch.no_grad():
            pred_depth, confidence, _ = self.forward(data)
        return pred_depth, confidence


class DepthModel(BaseDepthModel):
    def __init__(self, cfg, **kwards):
        super(DepthModel, self).__init__(cfg)
        model_type = cfg.model.type

    def inference(self, data):
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.forward(data)
        return pred_depth, confidence, output_dict


def get_monodepth_model(
        cfg: dict,
        **kwargs
) -> nn.Module:
    # config depth  model
    model = DepthModel(cfg, **kwargs)
    # model.init_weights(load_imagenet_model, imagenet_ckpt_fpath)
    assert isinstance(model, nn.Module)
    return model


def get_configured_monodepth_model(
        cfg: dict,
) -> nn.Module:
    """
        Args:
        @ configs: configures for the network.
        @ load_imagenet_model: whether to initialize from ImageNet-pretrained model.
        @ imagenet_ckpt_fpath: string representing path to file with weights to initialize model with.
        Returns:
        # model: depth model.
    """
    model = get_monodepth_model(cfg)
    return model


import logging
import os
def load_ckpt(load_path, model, optimizer=None, scheduler=None, strict_match=True, loss_scaler=None):
    """
    Load the check point for resuming training or finetuning.
    """
    logger = logging.getLogger()
    if os.path.isfile(load_path):

        logger.info(f"Loading weight '{load_path}'")
        checkpoint = torch.load(load_path, map_location="cpu")
        ckpt_state_dict  = checkpoint['model_state_dict']
        model.module.load_state_dict(ckpt_state_dict, strict=strict_match)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if loss_scaler is not None and 'scaler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scaler'])
        del ckpt_state_dict
        del checkpoint
        # if main_process():
        logger.info(f"Successfully loaded weight: '{load_path}'")
        if scheduler is not None and optimizer is not None:
            logger.info(f"Resume training from: '{load_path}'")
    else:
        # if main_process():
        raise RuntimeError(f"No weight found at '{load_path}'")
    return model, optimizer, scheduler, loss_scaler