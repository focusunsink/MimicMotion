import logging
import re

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    FluxAttnProcessor2_0,
)

import sys
import os


def quantize_lvl(model_id, backbone, quant_level=2.5, linear_only=False, enable_conv_3d=True):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if linear_only:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            else:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ) and name != "proj_out":  # Disable the final output layer from flux model
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, torch.nn.Conv3d) and not enable_conv_3d:
            """
                Error: Torch bug, ONNX export failed due to unknown kernel shape in QuantConv3d.
                TRT_FP8QuantizeLinear and TRT_FP8DequantizeLinear operations in UNetSpatioTemporalConditionModel for svd
                cause issues. Inputs on different devices (CUDA vs CPU) may contribute to the problem.
            """
            module.input_quantizer.disable()
            module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            # TRT only supports FP8 MHA with head_size % 16 == 0.
            head_size = int(module.inner_dim / module.heads)
            if quant_level >= 4 and head_size % 16 == 0:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
                if model_id.startswith("flux.1"):
                    if name.startswith("transformer_blocks"):
                        module.bmm2_output_quantizer.enable()
                    else:
                        module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", False)
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
                module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", True)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = os.path.dirname(os.path.abspath(current_dir))

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

data = torch.load("unet2_16frame_res448.pt")


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


import modelopt.torch.quantization as mtq

model = unet

def forward_loop(model):
    for i in range(100):
        model(latent_model_input, t, encoder_hidden_states, added_time_ids, pose_latents)


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|proj_out).*"
    )
    return pattern.match(name) is not None

def filter_func_no_proj_out(name): # used for Flux 
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None



def generate_fp8_scales(unet):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in unet.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax") and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (127 / 448.0)
        elif isinstance(module, Attention) and (
            hasattr(module.q_bmm_quantizer, "_amax") and module.q_bmm_quantizer is not None
        ):
            module.q_bmm_quantizer._num_bits = 8
            module.q_bmm_quantizer._amax = module.q_bmm_quantizer._amax * (127 / 448.0)
            module.k_bmm_quantizer._num_bits = 8
            module.k_bmm_quantizer._amax = module.k_bmm_quantizer._amax * (127 / 448.0)
            module.v_bmm_quantizer._num_bits = 8
            module.v_bmm_quantizer._amax = module.v_bmm_quantizer._amax * (127 / 448.0)
            module.softmax_quantizer._num_bits = 8
            module.softmax_quantizer._amax = module.softmax_quantizer._amax * (127 / 448.0)


model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop)

inputs=(latent_model_input, t, encoder_hidden_states, added_time_ids, pose_latents)
input_names=["latent_model_input", "timestep", "encoder_hidden_states", "added_time_ids", "pose_latents"]
onnx_path = "./models/onnx/unet_quant.onnx"
output_names = ["_noise_pred"]


torch.save(model.state_dict(), "./models/quant/unet_quantized_state_dict.pt")
import modelopt.torch.opt as mto
mto.save(model, "./models/quant/unet_quantized.pt")
onnx_inputs = [inp for inp in inputs]



quantize_lvl(1.5, model, 3.0)
mtq.disable_quantizer(model, filter_func)

generate_fp8_scales(model)

# from onnx_utils import ammo_export_sd
# ammo_export_sd(model, './', 'onnx_svd')
torch.cuda.empty_cache() 
print(model)
"""
https://github.com/pytorch/pytorch/issues/146591
"""
torch.onnx.export(model, tuple(onnx_inputs),
                  onnx_path=onnx_path,
                  opset_version=17,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  large_model=True,
                  verbose=True,
                  )

print("Done")
