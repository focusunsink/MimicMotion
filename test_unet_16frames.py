import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


import sys
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录加入 sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel


unet = UNetSpatioTemporalConditionModel.from_config(
                                                    UNetSpatioTemporalConditionModel.load_config("models/svd", subfolder="unet")
                                                    )

ckpt = torch.load("models/MimicMotion_1-1.pth")

unet_ckpt = {}

for k, v in ckpt.items():
    if k.startswith("unet"):
        unet_ckpt[k[5:]] = v


unet.load_state_dict(unet_ckpt)
unet.eval()
unet = unet.to("cuda:0").half()


latent_model_input = torch.zeros([1, 16, 8, 128, 72]).to("cuda:0").half()
t = torch.tensor(1.6377).to("cuda:0").half().reshape([1])
image_embeddings = torch.zeros([1, 1, 1024]).to("cuda:0").half()
added_time_ids = torch.zeros([1, 3]).to("cuda:0").half()

# unet = torch.compile(unet)
import time


with torch.no_grad():
    for i in range(10):
        torch.cuda.synchronize()
        st = time.time()
        _noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=image_embeddings,
                            added_time_ids=added_time_ids,
                        )[0]

        torch.cuda.synchronize()
        ed =  time.time()
        print("unet time used: ", ed - st)


print(_noise_pred)
print(_noise_pred.dtype)
print(_noise_pred.shape)
