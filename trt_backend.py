import torch
import torch.nn as nn
import tensorrt as trt
from collections import OrderedDict, namedtuple
import numpy as np
import time 

class ModelBackendBase():
    def __init__(self, engine_path, device):
        """
        
        """
        self.is_trt10 = int(trt.__version__.split(".")[0]) >= 10
        self.engine_path = engine_path

        self.device=device

      
        self._load_engine()
      

        import logging 
        self.LOGGER = logging.getLogger("Audio Logger")
        self.print_log = True
            


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
        self.batch_size = 1  
        return bindings, binding_addrs, output_names

    def _load_engine(self):
       
        logger = trt.Logger(trt.Logger.INFO)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        print("context created")
        self.bindings, self.binding_addrs, self.output_names = self._set_bindings(self.model, self.context)


   


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
        self.context.set_input_shape("latent_model_input", [1, 16, 8, 96, 56]) 
        self.bindings["latent_model_input"] = self.bindings["latent_model_input"]._replace(shape=[1, 16, 8, 96, 56])
        

        self.context.set_input_shape("t", [1]) 
        self.bindings["t"] = self.bindings["t"]._replace(shape=[1])
        
        self.context.set_input_shape("image_embeddings", [1, 1, 1024]) 
        self.bindings["image_embeddings"] = self.bindings["image_embeddings"]._replace(shape=[1, 1, 1024])
                
        self.context.set_input_shape("added_time_ids", [1, 3]) 
        self.bindings["added_time_ids"] = self.bindings["added_time_ids"]._replace(shape=[1, 3])
    
        self.context.set_input_shape("pose_latents", [16, 320, 96, 56])
        self.bindings["pose_latents"] = self.bindings["pose_latents"]._replace(shape=[16, 320, 96, 56])

        """set output shape"""
   
        self.bindings["_noise_pred"].data.resize_(tuple([1, 16, 4, 96, 56]))
    

    def set_addr(self, latent_model_input, t, image_embeddings, added_time_ids, pose_latents):
        """
        set input addr
        """
        self.binding_addrs["latent_model_input"] = int(latent_model_input.data_ptr())
        self.binding_addrs["t"] = int(t.data_ptr())
        self.binding_addrs["image_embeddings"] = int(image_embeddings.data_ptr())
        self.binding_addrs["added_time_ids"] = int(added_time_ids.data_ptr())
        self.binding_addrs["pose_latents"] = int(pose_latents.data_ptr())

   

   

    def run(self,  latent_model_input, t, image_embeddings, added_time_ids, pose_latents):
        
        self.set_shape()
        self.set_addr(latent_model_input, t, image_embeddings, added_time_ids, pose_latents)

        self.context.execute_v2(list(self.binding_addrs.values()))

        """ outputs """
        y = [self.bindings[x].data for x in self.output_names]

        
        return y[0].float()
