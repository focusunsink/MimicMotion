"""
Position id must be float32

"""

import torch
import torch.nn as nn
import tensorrt as trt
from collections import OrderedDict, namedtuple
import numpy as np
import time 

trt_to_torch_dtype_dict = {
    trt.DataType.BOOL     : torch.bool,
    trt.DataType.UINT8    : torch.uint8,
    trt.DataType.INT8     : torch.int8,
    trt.DataType.INT32    : torch.int32,
    trt.DataType.INT64    : torch.int64,
    trt.DataType.HALF     : torch.float16,
    trt.DataType.FLOAT    : torch.float32,
    trt.DataType.BF16     : torch.bfloat16
}


class ModelBackend():
    
    def __init__():
        pass

class ModelBackendBase():
    def __init__(self, engine_path, batch_size=1, device="cuda:0"):
        """
        
        """
        self.is_trt10 = int(trt.__version__.split(".")[0]) >= 10
        self.engine_path = engine_path
        self.device=device
        self.device_id = int(device[-1])
        self.batch_size = batch_size
        
        self.tensors = OrderedDict()

        self.latent_height = 96
        self.latent_width = 56
        self.frames = 16
      
        self._load_engine()
      
        self.allocate_buffers()
        import logging 
        self.LOGGER = logging.getLogger("Audio Logger")
        self.print_log = True
            

    def make_shape_dict(self):
        
        self.shape_dict = {
                'latent_model_input': (self.batch_size, self.frames, 8, self.latent_height,  self.latent_width),
                't': (1,),
                'image_embeddings': (self.batch_size, 1, 1024),
                'added_time_ids': (self.batch_size, 3),
                'pose_latents': (self.frames * self.batch_size, 320,  self.latent_height,  self.latent_width),
                '_noise_pred': (self.batch_size,  self.frames, 4,  self.latent_height,  self.latent_width)
                }


    def _set_bindings(self, model, context):
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        num = range(model.num_io_tensors)
        for i in num:
    
            
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print("inputs: name: ", name)
                if dtype == np.float16:
                    fp16 = True
            else:
                output_names.append(name)
            shape = tuple(context.get_tensor_shape(name))

          
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        return bindings, binding_addrs, output_names

    def _load_engine(self):
        import os
        from cuda import cudart
        torch.cuda.set_device(self.device_id) 
        # os.environ['CUDA_DEVICE'] = str(self.device[-1])
        # import pycuda.driver as cuda
        # import pycuda.autoinit
       
        logger = trt.Logger(trt.Logger.INFO)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.activate()
        
        status, self.stream = cudart.cudaStreamCreate()

        if status == cudart.cudaError_t.cudaSuccess:
            print(f"CUDA Stream 创建成功: {self.stream}")
        else:
            print(f"CUDA Stream 创建失败，错误代码: {self.status}")
    


    def activate(self):
        from cuda import cudart
        _, shared_device_memory = cudart.cudaMalloc(self.engine.device_memory_size)
        if shared_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = shared_device_memory
        else:
            self.context = self.engine.create_execution_context()
        
        self.bindings, self.binding_addrs, self.output_names = self._set_bindings(self.engine, self.context)


      

    def allocate_buffers(self):
        device = self.device
        self.make_shape_dict()
        shape_dict = self.shape_dict
         
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
                print(f"[W]: {self.engine_path}: Could not find '{name}' in shape dict {shape_dict}.  Using shape {shape} inferred from the engine.")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            dtype=trt_to_torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device)
            self.tensors[name] = tensor

    def _print_log(self, qid=0):
        if self.print_log:
            self.LOGGER.info(f"Query ID is {qid}, seq_len is {self.seq_len}")
            # ?~N??~O~V设?~G?~P~M称
            device_name = torch.cuda.get_device_name(self.device)
            self.LOGGER.info(f"Device Name: {device_name}")
            allocated_memory = torch.cuda.memory_allocated(self.device)
            cached_memory = torch.cuda.memory_reserved(self.device)

            self.LOGGER.info(f"Allocated Memory: {allocated_memory / (1024 ** 2)} MB")
            self.LOGGER.info(f"Cached Memory: {cached_memory / (1024 ** 2)} MB")


    def set_shape(self):
        
        """set input shape"""
        self.bindings["latent_model_input"] = self.bindings["latent_model_input"]._replace(shape=[self.batch_size, 16, 8, 96, 56])
        self.bindings["t"] = self.bindings["t"]._replace(shape=[1])
        self.bindings["image_embeddings"] = self.bindings["image_embeddings"]._replace(shape=[self.batch_size, 1, 1024])
        self.bindings["added_time_ids"] = self.bindings["added_time_ids"]._replace(shape=[self.batch_size, 3])
        self.bindings["pose_latents"] = self.bindings["pose_latents"]._replace(shape=[16 * self.batch_size, 320, 96, 56])

        """set output shape"""
   
        self.bindings["_noise_pred"].data.resize_(tuple([self.batch_size, 16, 4, 96, 56]))
    
    
    

    def set_addr(self, latent_model_input, t, image_embeddings, added_time_ids, pose_latents):
        """
        set input addr
        """
        self.binding_addrs["latent_model_input"] = int(latent_model_input.data_ptr())
        self.binding_addrs["t"] = int(t.data_ptr())
        self.binding_addrs["image_embeddings"] = int(image_embeddings.data_ptr())
        self.binding_addrs["added_time_ids"] = int(added_time_ids.data_ptr())
        self.binding_addrs["pose_latents"] = int(pose_latents.data_ptr())

           
        self.context.set_tensor_address("latent_model_input", self.binding_addrs["latent_model_input"])
        self.context.set_tensor_address("t", self.binding_addrs["t"])
        self.context.set_tensor_address("image_embeddings", self.binding_addrs["image_embeddings"])
        self.context.set_tensor_address("added_time_ids", self.binding_addrs["added_time_ids"])
        self.context.set_tensor_address("pose_latents", self.binding_addrs["pose_latents"])

        # print(f"当前 self.device: {self.device}, 当前 PyTorch 设备: {torch.cuda.current_device()}")
        # print("_noise_pred", self._noise_pred.shape, self._noise_pred.dtype, self._noise_pred.device)
        # assert self._noise_pred.is_cuda, "_noise_pred must be on GPU!"

        # self.binding_addrs["_noise_pred"] = self._noise_pred.data_ptr()
        # success = self.context.set_tensor_address("_noise_pred", self.binding_addrs["_noise_pred"])
        # assert success, f"Failed to bind _noise_pred at {hex(self.binding_addrs["_noise_pred"])}!"

   

    def run(self,  latent_model_input, t, image_embeddings, added_time_ids, pose_latents):
        torch.cuda.set_device(self.device_id)
      
        # self.set_shape()
        
        # self.set_addr(latent_model_input, t, image_embeddings, added_time_ids, pose_latents)
      
        self.tensors["latent_model_input"].copy_(latent_model_input)
        self.tensors["t"].copy_(t)
        self.tensors["image_embeddings"].copy_(image_embeddings)
        self.tensors["added_time_ids"].copy_(added_time_ids)
        self.tensors["pose_latents"].copy_(pose_latents)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        noerror = self.context.execute_async_v3(self.stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
            exit(0)
