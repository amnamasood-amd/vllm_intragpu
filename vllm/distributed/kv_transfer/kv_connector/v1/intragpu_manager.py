from multiprocessing.managers import BaseManager, SyncManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from typing import Any, Callable, Optional, Union

class GPUManager(SyncManager):
        pass

GPUManager.register("get_queue")

"""
def register_kvcachemanager(kv_role, kvcachemanager: Optional[KVCacheManager]):
        if kv_role=="kv_producer":
                assert kvcachemanager is not None
                GPUManager.register("get_kvcachemanager", callable=lambda: kvcachemanager)
        else:
                GPUManager.register("get_kvcachemanager")

"""