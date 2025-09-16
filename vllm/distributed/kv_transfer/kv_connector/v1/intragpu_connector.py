# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

#from vllm.distributed.kv_transfer.kv_connector.v1.intragpu_manager import GPUManager
from multiprocessing.managers import BaseManager, SyncManager
import multiprocessing
from vllm.v1.core.kv_cache_manager import KVCacheManager

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class SchedulerManager(SyncManager):
    pass

SchedulerManager.register("get_queue")
SchedulerManager.register("get_list")

class WorkerManager(SyncManager):
    pass

WorkerManager.register("get_queue")
WorkerManager.register("get_list")

@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(token_ids: list[int], block_ids: list[int], block_size: int,
                  is_store: bool, mm_hashes: list[str]) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )




@dataclass
class IntraGPUConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store,
                              mm_hashes))

class queue_manager:
    def __init__(self):
        #self.base_manager=SyncManager()
        #self.base_manager.start()
        self.q = multiprocessing.Queue()
        self.l = [] #self.base_manager.list([])
    #def set_list(self):
    #    self.l = self.base_manager.list([])
    def get_queue(self):
        return self.q
    def get_list(self):
        return self.l

class connector_manager:
    class custom_manager(BaseManager):
        pass

    def __init__(self, connector_role, kv_role):
        self.queue=multiprocessing.Queue()
        if kv_role=="kv_producer":
            self.custom_manager.register("get_queue", callable=lambda: self.queue)
        else:
            self.custom_manager.register("get_queue")
        authkey="secret"
        if connector_role == KVConnectorRole.SCHEDULER:
            logger.info("scheduler")
            self.gpu_manager=self.custom_manager(address=("127.0.0.1", 40001), authkey=authkey.encode("utf-8"))
        else:
            logger.info("worker")
            self.gpu_manager=self.custom_manager(address=("127.0.0.1", 40002), authkey=authkey.encode("utf-8"))
        
        if kv_role=="kv_producer":
            self.gpu_manager.start()
            self.queue.put("queue msg")
        else:
            self.gpu_manager.connect()
            self.queue = self.gpu_manager.get_queue()
            l=self.queue.get()
            print(l)
    



class IntraGPUConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class
    transfer_config=None

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        self.transfer_config = vllm_config.kv_transfer_config
        self._storage_path = self.transfer_config.get_from_extra_config(
            "shared_storage_path", "./tmpstorage")
        
        self.qmgr = queue_manager()
        self.running_prefill=[] #self.qmgr.base_manager.list([])
        self.gpu_manager=None

        #self.base_manager=SyncManager()
        """
        if self.transfer_config.kv_role=="kv_producer":
            GPUManager.register("get_queue", callable=lambda: self.qmgr.q)
        
        authkey="secret"
        if role == KVConnectorRole.SCHEDULER:
            logger.info("scheduler")
            self.gpu_manager=GPUManager(address=("127.0.0.1", 40001), authkey=authkey.encode("utf-8"))
        else:
            logger.info("worker")
            self.gpu_manager=GPUManager(address=("127.0.0.1", 40002), authkey=authkey.encode("utf-8"))
        if self.transfer_config.kv_role=="kv_producer":
            self.gpu_manager.start()
            self.qmgr.q.put("queue msg")
        else:
            self.gpu_manager.connect()
            #self.queue = self.gpu_manager.get_queue()
            l=self.qmgr.g.get()
            print(l)
        """
        """
        self.queue_manager = connector_manager(role, self.transfer_config.kv_role)
        """
        logger.info(vllm_config.kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

    #def get_queue(self):
    #    return self.queue
    
    #def get_running_prefill(self):
    #    return self.running_prefill

    def initialize_gpu_manager(self):
        #self.base_manager.start()
        #self.running_prefill = []#self.base_manager.list([4])
        if self.role==KVConnectorRole.SCHEDULER:
            GPUManager=SchedulerManager
        else:
            GPUManager=WorkerManager
        #self.qmgr.set_list()
        #hardcoding for now
        authkey="secret"
        if self.transfer_config.kv_role == "kv_producer":
            GPUManager.register("get_queue", callable= self.qmgr.get_queue)
            GPUManager.register("get_list", callable= self.qmgr.get_list)
        else:
            GPUManager.register("get_queue")
            GPUManager.register("get_list")
        
        if self.role==KVConnectorRole.SCHEDULER:
            self.gpu_manager = GPUManager(address=("127.0.0.1", 40001), authkey=authkey.encode("utf-8"))
        else:
            self.gpu_manager = GPUManager(address=("127.0.0.1", 40002), authkey=authkey.encode("utf-8"))
            
        if self.transfer_config.kv_role == "kv_producer":   
            logger.info("starting GPU manager producer")
            self.gpu_manager.start()
            self.qmgr.q.put("qmsg")
            
            self.running_prefill=self.qmgr.l
            logger.info("done starting GPU manager producer")
        elif self.transfer_config.kv_role == "kv_consumer":
            #GPUManager.register("get_kvcachemanager")
            logger.info("starting GPU manager consumer")
            self.gpu_manager.connect()
            self.qmgr.q=self.gpu_manager.get_queue()
            msg=self.qmgr.q.get()
            print(msg)
            self.running_prefill=self.gpu_manager.get_list()
            print(self.running_prefill)
            #l=q.get()
            #print(l)
            #print(self.gpu_manager.get_kvcachemanager().kv_cache_config)
    

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
        """
        attn_metadata = forward_context.attn_metadata

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            pass
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache 
                    layer. In shape [2, num_pages, page_size, xxx] if not 
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx] 
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape 
                    [num_tokens].
            """
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            """
            #if dst_kv_cache_layer != src_kv_cache:
            #print(dst_kv_cache_layersrc_kv_cache_layer)

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, IntraGPUConnectorMetadata)
        logger.info("Printing Connector metadata from scheduler")
        print(metadata)

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            logger.info("Printing request")
            print(request)
            if request.is_store:
                continue
            logger.info("Inject KV cache of %d tokens to the paged memory",
                        len(request.slot_mapping))
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE/MLP etc.
                kv_cache_attr = getattr(layer, 'kv_cache', None)
                if kv_cache_attr is None:
                    continue
                """
                kv_cache_layer = kv_cache_attr[ \
                        forward_context.virtual_engine]

                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes)
                kv_cache = safetensors.torch.load_file(
                    filename)["kv_cache"].cuda()
                inject_kv_into_layer(kv_cache_layer, kv_cache,
                                     request.slot_mapping)
                """

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping,
                                                                ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                               ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, IntraGPUConnectorMetadata)
        """
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes)
                kv_cache = extract_kv_from_layer(kv_layer,
                                                 request.slot_mapping)
                tensors = {"kv_cache": kv_cache.detach().cpu()}
                safetensors.torch.save_file(tensors, filename)
        """

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        if not self._found_match_for_request(request):
            return 0, False

        logger.info("External Cache Hit!")

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)

        return num_tokens_to_check - num_computed_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        pass
        #if num_external_tokens > 0:
        #    self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = IntraGPUConnectorMetadata()
        """
        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size,
                                 is_store=False,
                                 mm_hashes=new_req.mm_hashes)
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match_for_request(new_req):
                    meta.add_request(token_ids=new_req.prompt_token_ids,
                                     block_ids=new_req.block_ids[0],
                                     block_size=self._block_size,
                                     is_store=True,
                                     mm_hashes=new_req.mm_hashes)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = cached_reqs.resumed_from_preemption[i]

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                # NOTE(rob): cached_req_data does not have the full
                # list of token ids (only new tokens). So we look it
                # up in the actual request object.
                request = self._requests_need_load[req_id]
                total_tokens = num_computed_tokens + num_new_tokens
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = new_block_ids[0]

                meta.add_request(token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size,
                                 is_store=False,
                                 mm_hashes=request.mm_hashes)
                total_need_load += 1

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        """
        return meta
        

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """Check if the cache is hit for the request.
        """
        """
        num_tokens_to_check = align_to_block_size(
            len(request.prompt_token_ids) - 1, self._block_size)
        foldername = self._generate_foldername_debug(torch.tensor(
            request.prompt_token_ids)[:num_tokens_to_check],
                                                     request.mm_hashes,
                                                     create_folder=False)
        return os.path.exists(foldername)
        """
        return True

    def _generate_foldername_debug(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input 
        ids.
        """
        token_bytes = token_ids.numpy().tobytes()
        # Add mm_hashes to the bytes being hashed to avoid path traversal and
        # to create a canonical key.
        if mm_hashes:
            mm_str = "-".join(mm_hashes)
            token_bytes += mm_str.encode('utf-8')
        input_ids_hash = hashlib.md5(token_bytes,
                                     usedforsecurity=False).hexdigest()

        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> str:
        """Generate a file name based on the layer name and the hash 
        of the bytes of the input ids.
        """
        foldername = self._generate_foldername_debug(token_ids,
                                                     mm_hashes=mm_hashes,
                                                     create_folder=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size.
    """
    return (num_tokens - 1) // block_size * block_size
