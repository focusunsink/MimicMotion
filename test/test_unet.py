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

data = torch.load("unet_input_output.pt")


latent_model_input = data["latent_model_input"]
t = data["t"]
encoder_hidden_states = data["encoder_hidden_states"]
added_time_ids = data["added_time_ids"]
pose_latents = data["pose_latents"]
_noise_pred_gt = data["_noise_pred"]

print("latent_model_input", latent_model_input.shape, latent_model_input.dtype)
print("t", t.shape, t.dtype)
print("encoder_hidden_states", encoder_hidden_states.shape, encoder_hidden_states.dtype)
print("added_time_ids", added_time_ids.shape, added_time_ids.dtype)
print("pose_latents", pose_latents.shape, pose_latents.dtype)
print("_noise_pred_gt", _noise_pred_gt.shape, _noise_pred_gt.dtype)


#['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']

unet = torch.compile(unet)
import time


with torch.no_grad():
    for i in range(5):
        torch.cuda.synchronize()
        st = time.time()
        _noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            added_time_ids=added_time_ids,
                            pose_latents = pose_latents,
                        )[0]

        torch.cuda.synchronize()
        ed =  time.time()
        print("unet time used: ", ed - st)


print("simimlarity ", torch.cosine_similarity(_noise_pred_gt, _noise_pred))
err = torch.abs(_noise_pred - _noise_pred_gt)
sum_err = torch.sum(err)
#print("err ", err)
print("mean err", torch.mean(err))
print("sum err", sum_err)
print("max err", torch.max(err))


from trt_backend import ModelBackendBase


trt_backend = ModelBackendBase("./engines/unet2_16f_448res.engine", "cuda:0")


import time

for i in range(10):
    torch.cuda.synchronize()
    st = time.time()
    trt_output = trt_backend.run(latent_model_input, t, encoder_hidden_states, added_time_ids, pose_latents)
    torch.cuda.synchronize()
    ed = time.time()
    print("infer time used: ", ed - st)



print("simimlarity ", torch.cosine_similarity(_noise_pred_gt, trt_output))
err = torch.abs(trt_output - _noise_pred_gt)
sum_err = torch.sum(err)
#print("err ", err)
print("sum err", sum_err)
print("mean err", torch.mean(err))
print("max error: ", torch.max(err))
