from cfg import cfg
from tools import get_configured_monodepth_model
from tools import load_ckpt
import torch

model = get_configured_monodepth_model(cfg)

ckpt_path='convlarge_hourglass_0.3_150_step750k_v1.1.pth'
# model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(ckpt_path, map_location="cpu")
ckpt_state_dict = checkpoint['model_state_dict']
model.load_state_dict(ckpt_state_dict, strict=True)
# model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
model.eval()

print(model)