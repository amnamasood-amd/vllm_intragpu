# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata)
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: list[int]
    mm_kwargs: list[MultiModalKwargsItem]
    mm_hashes: list[str]
    mm_positions: list[PlaceholderRange]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_kwargs=request.mm_kwargs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
        )

    @classmethod
    def from_newrequestdataprefill(
        cls,
        request: NewRequestDataPrefill,
    ) -> NewRequestData:
        return cls(
            req_id=request.req_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_kwargs=[],
            mm_hashes=[],
            mm_positions=[],
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=request.block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=None,
        )

    def __repr__(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids={self.prompt_token_ids},"
                f"mm_kwargs={self.mm_kwargs},"
                f"mm_hashes={self.mm_hashes},"
                f"mm_positions={self.mm_positions},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request}"
                ")")

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids_len={len(self.prompt_token_ids)},"
                f"mm_kwargs={self.mm_kwargs},"
                f"mm_hashes={self.mm_hashes},"
                f"mm_positions={self.mm_positions},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                f"lora_request={self.lora_request}"
                ")")


@dataclass
class CachedRequestData:

    req_ids: list[str]
    # If resumed_from_preemption is False, new_block_ids will be appended to
    # the request's block IDs. If True, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_from_preemption: list[bool]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    new_block_ids: list[Optional[tuple[list[int], ...]]]
    num_computed_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        )


@dataclass
class SchedulerOutput:

    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Dict of request ids to their index within the batch
    # for filling the next token bitmask
    structured_output_request_ids: dict[str, int]
    # the bitmask for the whole batch
    grammar_bitmask: Optional[npt.NDArray[np.int32]]

    # KV Cache Connector metadata.
    kv_connector_metadata: Optional[KVConnectorMetadata] = None

    @classmethod
    def from_scheduleroutputprefill (
        cls,
        scheduleroutputprefill: SchedulerOutputPrefill,
    ) -> SchedulerOutput:
        newrequests = [NewRequestData.from_newrequestdataprefill(req) for req in scheduleroutputprefill.scheduled_new_reqs]
        return cls(
            scheduled_new_reqs=newrequests,
            scheduled_cached_reqs=scheduleroutputprefill.scheduled_cached_reqs,
            num_scheduled_tokens=scheduleroutputprefill.num_scheduled_tokens,
            total_num_scheduled_tokens= scheduleroutputprefill.total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=scheduleroutputprefill.num_common_prefix_blocks,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
            kv_connector_metadata=None,
        )


@dataclass
class NewRequestDataPrefill:

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: Optional[SamplingParams]
    pooling_params: Optional[PoolingParams]
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> NewRequestDataPrefill:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )

    @classmethod
    def from_newrequestdata(
        cls,
        request: NewRequestData,
    ) -> NewRequestDataPrefill:
        return cls(
            req_id=request.req_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=request.block_ids,
            num_computed_tokens=request.num_computed_tokens,
        )

    def __repr__(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids={self.prompt_token_ids},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                ")")

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self):
        return (f"NewRequestData("
                f"req_id={self.req_id},"
                f"prompt_token_ids_len={len(self.prompt_token_ids)},"
                f"sampling_params={self.sampling_params},"
                f"block_ids={self.block_ids},"
                f"num_computed_tokens={self.num_computed_tokens},"
                ")")

@dataclass
class SchedulerOutputPrefill:

    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestDataPrefill]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData
    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]


    @classmethod
    def from_scheduler_output(
        cls,
        scheduler_output: SchedulerOutput,
    ) -> SchedulerOutputPrefill:
        new_requests = [NewRequestDataPrefill.from_newrequestdata(new_req) for new_req in scheduler_output.scheduled_new_reqs]
        return cls(
            scheduled_new_reqs=new_requests,
            scheduled_cached_reqs=scheduler_output.scheduled_cached_reqs,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_common_prefix_blocks=scheduler_output.num_common_prefix_blocks,
            finished_req_ids=scheduler_output.finished_req_ids,
        )
        
