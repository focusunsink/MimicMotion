"""
position ids must be float32
"""
import sys 
import torch
import numpy as np

import tensorrt as trt
import logging 
LOGGER = logging.getLogger("Audio Logger")
import argparse 

import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import os 



class BuilderManager(object):
    def __init__(self, model, inputs, input_names, output_names, trt_model_dtype="fp32"):
        self.inputs = inputs
        self.input_names = input_names
        self.output_names = output_names
        self.torch_model = model
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.dynamic=False
        self.trt_model_dtype=trt_model_dtype
        self.is_trt10 = int(trt.__version__.split(".")[0]) >= 10
       

        print("Builder Manager Inited")
        print("trt version is ", trt.__version__)

    def _set_profile_shape(self, profile):
        
        profile.set_shape(
            "latent_model_input", 
            ([1, 16, 8, 96, 56]), ([1, 16, 8, 96, 56]), ([1, 16, 8, 96,56])
            )      
        profile.set_shape(
            "t",
            ([1]), ([1]), ([1])
            )            
        profile.set_shape(
            "image_embeddings", 
            ([1, 1, 1024]), ([1, 1, 1024]), ([1, 1, 1024])
            )
        profile.set_shape(
            "pose_latents",
            ([16, 320, 96, 56]), ([16, 320, 96, 56]), ([16, 320, 96, 56])
                )
        profile.set_shape(
            "added_time_ids", 
            ([1, 3]), ([1, 3]), ([1, 3])
            ) 

        return profile

    
    
    def to_onnx(self, onnx_path, simplify_flag=False): 
        import onnx 
        self.onnx_path = onnx_path
        external_data_folder = "./"
        print("onnx_path", onnx_path)
        def location(tensor_name): 
            import os  
            # 您可以根据tensor_name或其他逻辑来生成文件名  
            filename = os.path.join(external_data_folder, f"{tensor_name}.bin")  
            return filename
    
        torch.onnx.export(self.torch_model , 
            self.inputs, 
            self.onnx_path, 
            input_names=self.input_names, 
            output_names=self.output_names, 
            verbose=True,
            opset_version=17,
            do_constant_folding=True,
            )
        
        if False: 
            from onnxsim import simplify
            model_onnx = onnx.load(self.onnx_path)  # load onnx model
            print(model_onnx.ir_version)
            model_onnx.ir_version = 7
            onnx.save_model(model_onnx, self.onnx_path)
            model_onnx = onnx.load(self.onnx_path)
            onnx_model_simp, check = simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
            print("onnx simplify convert success!")
            onnx.save_model(onnx_model_simp, self.onnx_path)
            self.onnx_model = onnx_model_simp

        print("Convert to Onnx Done")
    
    def _set_config(self, builder, network):
        config = builder.create_builder_config()
        workspace=2
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    
    
        # pass
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.set_flag(trt.BuilderFlag.MONITOR_MEMORY)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        print("network.num_layers", network.num_layers)
        
        #if self.trt_model_dtype == "fp16":
            # pass
            # config.set_flag(trt.BuilderFlag.FP16)
            # print("network.num_layers", network.num_layers)
      
            # for i in range(network.num_layers):
            #     tmp_layer = network.get_layer(i)
            #     if "norm" in tmp_layer.name and "Cast_1" not in tmp_layer.name and "Mul_1" not in tmp_layer.name : #== "/ar_decoder/norm/Mul":
            #         print("setting ", tmp_layer.name, " precision")
            #         network.get_layer(i).precision = trt.DataType.FLOAT
            #         network.get_layer(i).set_output_type(0, trt.DataType.FLOAT)
            #     print(tmp_layer.name, tmp_layer.type, tmp_layer.num_inputs, tmp_layer.precision, tmp_layer.precision_is_set)
            #config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        trt.tactic_source = trt.TacticSource.CUBLAS
        config.builder_optimization_level = 5
        config.max_num_tactics = 2

        return config
    



    def build_engine(self, engine_path):

        trt.init_libnvinfer_plugins(None, '')
        
        """1. Logger"""
        logger = trt.Logger(trt.Logger.VERBOSE)
        trt_engine_file = engine_path
       
        verbose=True
        # if verbose:
        #     logger.min_severity = trt.Logger.Severity.VERBOSE
        """2. builder"""
        builder = trt.Builder(logger)
       
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        """3. network"""
        #flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flag)
        print("network created")
        """4. parse """
        with trt.OnnxParser(network, logger) as parser:
            print(self.onnx_path)
            parser.parse_from_file(self.onnx_path)
          
        """5. log bindings """
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'Tensorrt input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'Tensorrt output "{out.name}" with shape{out.shape} {out.dtype}')

        """6 profile """
        profile = builder.create_optimization_profile()
        profile = self._set_profile_shape(profile)

        """7 config """
        config = self._set_config(builder, network)
        config.add_optimization_profile(profile)
        
        """8. build engine"""            
        build = builder.build_serialized_network
        print(builder)
        print(trt_engine_file)
        engine = build(network, config) 
        if engine is None:
            sys.exit("Failed building engine")

        with open(trt_engine_file, "wb") as f:
            f.write(engine)

            print("Succeeded building engine")
        
        # """insepct"""
        print("inspecting")
        with open(trt_engine_file, "rb") as f:
            isp_engine=trt.Runtime(logger).deserialize_cuda_engine(f.read())
            inspector = isp_engine.create_engine_inspector()

        print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

        return engine

if __name__ == "__main__":
    import sys 
    device = "cuda:0"
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    # 将当前目录加入 sys.path
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    print(sys.path)

    from mimicmotion.modules.unet_v2 import UNetSpatioTemporalConditionModel


    unet = UNetSpatioTemporalConditionModel.from_config(
                                                        UNetSpatioTemporalConditionModel.load_config("models/svd", subfolder="unet")
                                                        )

    ckpt = torch.load("models/MimicMotion_1-1.pth", weights_only=True)

    unet_ckpt = {}

    for k, v in ckpt.items():
        if k.startswith("unet"):
            unet_ckpt[k[5:]] = v


    unet.load_state_dict(unet_ckpt)
    unet.eval()
    unet = unet.to("cuda:0").half()


    latent_model_input = torch.zeros([1, 16, 8, 96, 56]).to("cuda:0").half()
    t = torch.tensor(1.6377).to("cuda:0").half()
    print(t.shape)
 
    t = t[None]
    print(t.shape)
    image_embeddings = torch.zeros([1, 1, 1024]).to("cuda:0").half()
    added_time_ids = torch.zeros([1, 3]).to("cuda:0").half()
    pose_latents = torch.zeros([16, 320, 96, 56]).to("cuda:0").half()
    model = unet

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.requires_grad = False 


    """
    engine
    """
    output_names = ["_noise_pred"]
   
    trt_builder = BuilderManager(
        model, 
        inputs=(latent_model_input, t, image_embeddings, added_time_ids, pose_latents),
        input_names=["latent_model_input", "t","image_embeddings", "added_time_ids", "pose_latents"],
        output_names=output_names,
        trt_model_dtype="fp16"
        )
    engine_path = "./engines/svd.engine"
    trt_builder.to_onnx(engine_path.replace(".engine", ".onnx"), simplify_flag=True)
    print("to onnx done. start to build engine")
    engine = trt_builder.build_engine(engine_path)
