from cfg import cfg
from tools import get_configured_monodepth_model


model = get_configured_monodepth_model(cfg)
print(model)