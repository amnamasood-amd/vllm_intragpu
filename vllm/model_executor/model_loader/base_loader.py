# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype, send_tensor_with_untyped_storage)
import multiprocessing
import torch.multiprocessing as mp
import pickle
from vllm.distributed.parallel_state import get_world_group

logger = init_logger(__name__)



class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config
        self.param_storage_list=[]

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig, connector_role=None, connector_q=None) -> nn.Module:
        """Load a model with the given configurations."""
        logger.info("In load_model with connector_role %s", connector_role)
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else \
                      load_config.device
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            #print(get_kv_transfer_group())
            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            if connector_role == None:
                self.load_weights(model, model_config)
            elif connector_role == "kv_producer":
                #kv producer/prefill instance is going to load weights and populate param_storage_list
                self.load_weights(model, model_config)
                assert connector_q is not None
                for param in model.parameters():
                    send_tensor_with_untyped_storage(param, self.param_storage_list)
                rank = get_world_group().local_rank
                with open("model_handles_"+str(rank)+".pkl",'wb') as file:
                    pickle.dump(self.param_storage_list, file)
                #connector_q.put(self.param_storage_list)
                #print(len(self.param_storage_list))
            else:
                assert connector_q is not None
                logger.info("getting parameter list")
                #self.param_storage_list = connector_q.get()
                rank = get_world_group().local_rank
                with open("model_handles_"+str(rank)+".pkl",'rb') as file:
                    self.param_storage_list = pickle.load(file)
                logger.info("length of param_storage_list %d", len(self.param_storage_list))
                weights=[]
                for spec in self.param_storage_list:
                    weights.append(mp.reductions.rebuild_cuda_tensor(**spec))
                i=0
                for param in model.parameters():
                    param.data = weights[i]
                    i+=1
                #self.load_weights(model, model_config)

                
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()
