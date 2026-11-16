#!/usr/bin/env python3
"""Move extract_hidden_states off the NPU hybrid KV pool.

Hidden states stay bf16 and keep all Eagle aux layers. They are copied
DtoH on TP0 each step into per-request CPU pages; request_finished writes
safetensors from those pages. MLA/indexer KV stays on NPU.

When the engine prefix cache is on, TP0 addresses Host HS by the same
BlockPool block_ids as NPU MLA KV: cpu_hs[num_gpu_blocks, block_size,
layers, hidden]. Prefix hits gather those ids; reuse of a block_id
overwrites the CPU slot. Hash LRU is not the prefix source of truth.
The table is unused whenever enable_prefix_caching is false (full DtoH).

Aligned with sunny-infra/vllm-sunny feat/hs-cpu-pages (V1) and
feat/hs-cpu-pages-mrv2 (V2): prefix flag latched in register_kv_caches,
NaN sentinels, save_hs_batch_tokens for MR V2, silent KV-xfer channel
in CPU-pages mode, then HS_CPU_BLOCK_TABLE for BlockPool id alignment.

Safe to re-run: files with HS_CPU_WORKER_REFUSE are skipped. Older
HS_CPU_PAGES / HS_CPU_PREFIX_HASH / HS_CPU_UPSTREAM / HS_CPU_BLOCK_TABLE
installs are upgraded in place. Hole checks run on the Worker (which
owns `_cpu_pages`); EngineCore request_finished only returns the path.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

MARKER = "HS_CPU_PAGES"
PREFIX_MARKER = "HS_CPU_PREFIX_HASH"
UPSTREAM_MARKER = "HS_CPU_UPSTREAM"
BLOCK_TABLE_MARKER = "HS_CPU_BLOCK_TABLE"
WORKER_REFUSE_MARKER = "HS_CPU_WORKER_REFUSE"

SITE = Path("/usr/local/python3.11.15/lib/python3.11/site-packages")

DEFAULTS = {
    "model_runner": SITE / "vllm_ascend/worker/model_runner_v1.py",
    "cache_only": SITE / "vllm/model_executor/models/extract_hidden_states.py",
    "proposer": SITE / "vllm/v1/spec_decode/extract_hidden_states.py",
    "connector": SITE
    / "vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py",
    "kv_transfer": SITE / "vllm/config/kv_transfer.py",
    "gpu_model_runner": SITE / "vllm/v1/worker/gpu_model_runner.py",
    "attn_utils": SITE / "vllm/v1/worker/gpu/attn_utils.py",
    "v2_speculator": SITE / "vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py",
    "v2_base_speculator": SITE / "vllm/v1/worker/gpu/spec_decode/speculator.py",
    "v2_init_speculator": SITE / "vllm/v1/worker/gpu/spec_decode/__init__.py",
    "v2_model_runner": SITE / "vllm/v1/worker/gpu/model_runner.py",
}

# Optional files are skipped (not an error) when a tree has no MR V2 / GPU runner.
OPTIONAL_KEYS = {
    "kv_transfer",
    "gpu_model_runner",
    "attn_utils",
    "v2_speculator",
    "v2_base_speculator",
    "v2_init_speculator",
    "v2_model_runner",
}


def _backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, bak)
    return bak


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def _replace_span(text: str, start: str, end: str, new: str, label: str) -> str:
    i = text.find(start)
    if i < 0:
        raise ValueError(f"{label}: start marker not found")
    j = text.find(end, i)
    if j < 0:
        raise ValueError(f"{label}: end marker not found")
    j += len(end)
    if text.find(start, i + 1) != -1:
        raise ValueError(f"{label}: start marker not unique")
    return text[:i] + new + text[j:]


def patch_model_runner(text: str) -> str:
    old = """            elif isinstance(attn_module, CacheOnlyAttentionLayer):
                # Only CacheOnlyAttentionLayer (extract_hidden_states draft model)
                # is handled here.
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
"""
    new = """            elif isinstance(attn_module, CacheOnlyAttentionLayer):
                # Only CacheOnlyAttentionLayer (extract_hidden_states draft model)
                # is handled here.
                # HS_CPU_PAGES: do not put HiddenStateCacheSpec into the NPU
                # hybrid pool. TP0 keeps HS on CPU; other ranks store nothing.
                _kvt = self.vllm_config.kv_transfer_config
                _hs_store = "cpu"
                if _kvt is not None:
                    _hs_store = str(
                        _kvt.get_from_extra_config("hs_store_device", "cpu")
                    ).lower()
                if _hs_store == "cpu":
                    logger.info(
                        "HS_CPU_PAGES: skip NPU kv_cache_spec for %s "
                        "(hidden states stored on TP0 CPU)",
                        layer_name,
                    )
                    continue
                if spec := attn_module.get_kv_cache_spec(self.vllm_config):
"""
    return _replace_once(text, old, new, "model_runner CacheOnly branch")


def patch_cache_only(text: str) -> str:
    old = """        output = torch.empty(0, device=to_cache.device, dtype=to_cache.dtype)

        # Note: dummy_out is used to force torch.compile to preserve ordering between
        # cache update and attention op (which triggers kv_connector transfer)
        dummy_out = unified_kv_cache_update(to_cache, self.layer_name)

        # Triggers kv_connector transfer via decorator
        _ = dummy_attention(self.layer_name, dummy_out)

        return output
"""
    new = """        output = torch.empty(0, device=to_cache.device, dtype=to_cache.dtype)

        # HS_CPU_PAGES: empty placeholder kv_cache means this layer is not in
        # the NPU pool; proposer already streamed to_cache to TP0 CPU pages.
        if self.kv_cache.numel() == 0:
            return output

        # Note: dummy_out is used to force torch.compile to preserve ordering between
        # cache update and attention op (which triggers kv_connector transfer)
        dummy_out = unified_kv_cache_update(to_cache, self.layer_name)

        # Triggers kv_connector transfer via decorator
        _ = dummy_attention(self.layer_name, dummy_out)

        return output
"""
    return _replace_once(text, old, new, "CacheOnlyAttentionLayer.forward")


def patch_proposer(text: str) -> str:
    old_init = """    def __init__(self, vllm_config: VllmConfig, device):
        assert vllm_config.speculative_config is not None
"""
    new_init = """    def __init__(self, vllm_config: VllmConfig, device, runner=None):
        assert vllm_config.speculative_config is not None
        # HS_CPU_UPSTREAM: keep a runner set by an Ascend subclass before super().
        self.runner = runner if runner is not None else getattr(self, "runner", None)
"""
    text = _replace_once(text, old_init, new_init, "proposer.__init__ runner")

    old_rank = """        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.eplb_state: EplbState | None = None
"""
    new_rank = """        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        kvt = vllm_config.kv_transfer_config
        store = "cpu"
        if kvt is not None:
            store = str(kvt.get_from_extra_config("hs_store_device", "cpu")).lower()
        self._hs_store_cpu = store == "cpu"

        self.eplb_state: EplbState | None = None
"""
    text = _replace_once(text, old_rank, new_rank, "proposer._hs_store_cpu")

    old_validate = '''        raise ValueError(f"Cache-only layer {layer!r} not in any KV cache group")
'''
    new_validate = '''        # HS_CPU_PAGES: layer is omitted from NPU groups; reuse group 0
        # common_attn_metadata (query_start_loc / num_computed_tokens).
        if getattr(self, "_hs_store_cpu", False) and kv_cache_config.kv_cache_groups:
            self.kv_cache_gid = 0
            return
        kvt = self.vllm_config.kv_transfer_config
        store = "cpu"
        if kvt is not None:
            store = str(kvt.get_from_extra_config("hs_store_device", "cpu")).lower()
        if store == "cpu" and kv_cache_config.kv_cache_groups:
            self.kv_cache_gid = 0
            return
        raise ValueError(f"Cache-only layer {layer!r} not in any KV cache group")
'''
    text = _replace_once(
        text, old_validate, new_validate, "validate_same_kv_cache_group"
    )

    old_propose = """        # Copy hidden states to buffer
        self.hidden_states[:num_tokens] = stacked_hidden_states

        assert self.attn_metadata_builder is not None
"""
    new_propose = _PROPOSER_CPU_STREAM
    return _replace_once(text, old_propose, new_propose, "proposer.propose CPU DtoH")


_PROPOSER_CPU_STREAM = '''        # Copy hidden states to buffer
        self.hidden_states[:num_tokens] = stacked_hidden_states

        # HS_CPU_UPSTREAM: stream this step to TP0 CPU pages and skip the
        # cache-only device write. Fail loudly: a silent skip leaves empty dumps.
        kvt = self.vllm_config.kv_transfer_config
        store = "cpu"
        if kvt is not None:
            store = str(kvt.get_from_extra_config("hs_store_device", "cpu")).lower()
        if store == "cpu" or getattr(self, "_hs_store_cpu", False):
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )

            if not has_kv_transfer_group():
                raise ValueError(
                    "hs_store_device=cpu requires an active KV transfer "
                    "group to stream hidden states to CPU pages"
                )
            connector = get_kv_transfer_group()
            save_fn = getattr(connector, "save_hs_batch", None)
            if save_fn is None:
                raise ValueError(
                    "hs_store_device=cpu requires a KV connector that "
                    "implements save_hs_batch (e.g. "
                    f"ExampleHiddenStatesConnector), got "
                    f"{type(connector).__name__}"
                )
            runner = getattr(self, "runner", None)
            if runner is None:
                raise ValueError(
                    "hs_store_device=cpu requires the drafter to hold a "
                    "model runner reference to resolve per-request ids"
                )
            if common_attn_metadata is None:
                raise ValueError(
                    "hs_store_device=cpu requires common_attn_metadata"
                )
            n_req = int(common_attn_metadata.num_reqs)
            save_fn(
                self.hidden_states[:num_tokens],
                common_attn_metadata,
                list(runner.input_batch.req_ids[:n_req]),
                getattr(runner, "requests", None),
            )
            return sampled_token_ids[:, :1]

        assert self.attn_metadata_builder is not None
'''


def patch_connector(text: str) -> str:
    old_pending = """@dataclass
class PendingSave:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    block_ids: list[int]
"""
    new_pending = """@dataclass
class PendingSave:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    block_ids: list[int]
    discard: bool = False  # HS_CPU_PAGES: abort without writing
"""
    text = _replace_once(text, old_pending, new_pending, "PendingSave")

    old_init = """        # Read the hidden-states group and its block size from the group spec;
        # cache_config.block_size is bumped (wrong) for hybrid verifiers.
        self._cache_kv_group_id = self._find_cache_kv_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config, kv_cache_config, self._cache_kv_group_id
        )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []  # set by self.register_kv_caches
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)
"""
    new_init = """        # Read the hidden-states group and its block size from the group spec;
        # cache_config.block_size is bumped (wrong) for hybrid verifiers.
        # HS_CPU_PAGES: CPU store does not register a HiddenStateCacheSpec group.
        self._hs_store_cpu = str(
            self._kv_transfer_config.get_from_extra_config("hs_store_device", "cpu")
        ).lower() == "cpu"
        if self._hs_store_cpu:
            self._cache_kv_group_id = 0
            self._block_size = vllm_config.cache_config.block_size
        else:
            self._cache_kv_group_id = self._find_cache_kv_group_id(kv_cache_config)
            self._block_size = self._get_cache_block_size(
                vllm_config, kv_cache_config, self._cache_kv_group_id
            )
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []  # set by self.register_kv_caches
        self._cpu_pages: dict[str, torch.Tensor] = {}
        self._hs_staging: torch.Tensor | None = None
        from collections import OrderedDict
        cc = vllm_config.cache_config
        self._hs_prefix_index_on = bool(
            self._hs_store_cpu and getattr(cc, "enable_prefix_caching", False)
        )
        self._hs_hash_blocks: OrderedDict = OrderedDict()
        self._hs_hash_tokens = 0
        self._hs_prefix_max_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_prefix_cache_max_tokens", 131072
            )
        )
        self._req_block_hashes: dict[str, list] = {}
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)
        if self._hs_store_cpu:
            logger.info(
                "HS_CPU_PAGES: hidden states stored on TP0 CPU, not in NPU KV"
            )
            if self._hs_prefix_index_on:
                logger.info(
                    "HS_CPU_PREFIX_HASH: enabled max_tokens=%s block_size=%s",
                    self._hs_prefix_max_tokens,
                    self._block_size,
                )
            else:
                logger.info(
                    "HS_CPU_PREFIX_HASH: disabled (engine prefix cache off)"
                )
"""
    text = _replace_once(text, old_init, new_init, "connector.__init__")

    old_tp_rank = """        # Only TP rank 0 writes hidden states to disk; other TP ranks no-op.
        # Set in register_kv_caches (after distributed init).
        self._is_tp_rank_zero: bool = True
"""
    new_tp_rank = """        # Only TP rank 0 writes hidden states to disk; other TP ranks no-op.
        # HS_CPU_PAGES: resolve rank here so save_hs_batch is safe before
        # register_kv_caches; register_kv_caches still refreshes it.
        try:
            self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0
        except Exception:
            self._is_tp_rank_zero = True
"""
    text = _replace_once(text, old_tp_rank, new_tp_rank, "connector tp rank")

    old_register = """    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Delay tp rank0 initialization until after distributed init
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )
"""
    new_register = """    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Delay tp rank0 initialization until after distributed init
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0

        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: no NPU HiddenStateCacheSpec tensor to bind.
            self.cache_layers = []
            self._kv_cache = None
            logger.info(
                "HS_CPU_PAGES: register_kv_caches skipped NPU HS group (tp0=%s)",
                self._is_tp_rank_zero,
            )
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )
"""
    text = _replace_once(text, old_register, new_register, "register_kv_caches")

    old_write = """    @staticmethod
    def _write_tensors(
"""
    new_write = '''    def _hs_prefix_enabled(self) -> bool:
        return bool(
            getattr(self, "_hs_store_cpu", False)
            and getattr(self, "_hs_prefix_index_on", False)
            and getattr(self, "_is_tp_rank_zero", False)
        )

    def _block_extra_keys(self, req_state):
        extra = []
        lora = getattr(req_state, "lora_request", None)
        if lora is not None:
            extra.append(getattr(lora, "lora_name", None) or str(lora))
        return tuple(extra) if extra else None

    def _ensure_req_hashes(self, req_id, requests, n_blocks):
        hashes = self._req_block_hashes.setdefault(req_id, [])
        if n_blocks <= 0 or len(hashes) >= n_blocks:
            return hashes[: max(n_blocks, 0)]
        if requests is None or req_id not in requests:
            return hashes
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core import kv_cache_utils as _kvu
        from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

        st = requests[req_id]
        ids = list(st.prompt_token_ids or [])
        ids.extend(st.output_token_ids or [])
        bs = int(self._block_size)
        algo = getattr(
            self._vllm_config.cache_config, "prefix_caching_hash_algo", "sha256"
        )
        hash_fn = get_hash_fn_by_name(algo)
        if getattr(_kvu, "NONE_HASH", None) is None:
            init_none_hash(hash_fn)
        extra = self._block_extra_keys(st)
        while len(hashes) < n_blocks:
            i = len(hashes)
            lo = i * bs
            hi = lo + bs
            if hi > len(ids):
                break
            prev = hashes[-1] if hashes else None
            hashes.append(hash_block_tokens(hash_fn, prev, ids[lo:hi], extra))
        return hashes

    def _evict_hs_hash_blocks(self) -> None:
        max_tok = int(getattr(self, "_hs_prefix_max_tokens", 0) or 0)
        blocks = getattr(self, "_hs_hash_blocks", None)
        if blocks is None or max_tok <= 0:
            return
        while self._hs_hash_tokens > max_tok and blocks:
            _, old = blocks.popitem(last=False)
            self._hs_hash_tokens -= int(old.shape[0])

    def _fill_prefix_from_hash(self, req_id, page, n_prefix, requests) -> int:
        # HS_CPU_PREFIX_HASH: copy cached full blocks into a new request page.
        if not self._hs_prefix_enabled() or n_prefix <= 0:
            return 0
        bs = int(self._block_size)
        n_blocks = n_prefix // bs
        if n_blocks <= 0:
            return 0
        hashes = self._ensure_req_hashes(req_id, requests, n_blocks)
        copied = 0
        for i, h in enumerate(hashes[:n_blocks]):
            blk = self._hs_hash_blocks.get(h)
            if blk is None:
                logger.warning(
                    "HS_CPU_PREFIX_HASH: miss req_id=%s block=%s/%s",
                    req_id,
                    i,
                    n_blocks,
                )
                break
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi or blk.shape[0] != bs:
                break
            page[lo:hi].copy_(blk)
            self._hs_hash_blocks.move_to_end(h)
            copied += 1
        if copied:
            logger.info(
                "HS_CPU_PREFIX_HASH: filled %s/%s prefix blocks req_id=%s",
                copied,
                n_blocks,
                req_id,
            )
        return copied * bs

    def _publish_full_blocks(self, req_id, page, start, need, requests) -> None:
        if not self._hs_prefix_enabled() or need <= start:
            return
        bs = int(self._block_size)
        first = start // bs
        last = need // bs
        if last <= first:
            return
        hashes = self._ensure_req_hashes(req_id, requests, last)
        for i in range(first, min(last, len(hashes))):
            h = hashes[i]
            if h in self._hs_hash_blocks:
                self._hs_hash_blocks.move_to_end(h)
                continue
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi:
                break
            self._hs_hash_blocks[h] = page[lo:hi].contiguous().clone()
            self._hs_hash_tokens += bs
        self._evict_hs_hash_blocks()

    def save_hs_batch(
        self,
        hidden_states: torch.Tensor,
        common_attn_metadata,
        req_ids: list[str] | None,
        requests=None,
    ) -> None:
        """DtoH this step's hidden states into per-request CPU pages (TP0).

        HS_CPU_PAGES. ``hidden_states`` is [num_tokens, num_layers, hidden].
        Logical write offset is ``num_computed_tokens`` so chunked prefill
        concatenates. HS_CPU_PREFIX_HASH: a new request with prefix-cache
        hits copies cached CPU blocks into [0, num_computed_tokens).
        """
        if not getattr(self, "_hs_store_cpu", False):
            return
        if not self._is_tp_rank_zero:
            return
        if req_ids is None or common_attn_metadata is None:
            return
        if hidden_states.numel() == 0:
            return

        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            return

        qsl = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if qsl is None:
            qsl = common_attn_metadata.query_start_loc
        if hasattr(qsl, "detach"):
            qsl = qsl.detach()
        if hasattr(qsl, "cpu") and getattr(qsl, "is_cuda", False):
            qsl = qsl.cpu()
        qsl_list = qsl[: num_reqs + 1].tolist()

        nct = getattr(common_attn_metadata, "num_computed_tokens_cpu", None)
        if nct is not None:
            if hasattr(nct, "detach"):
                nct = nct.detach()
            if hasattr(nct, "cpu") and getattr(nct, "is_cuda", False):
                nct = nct.cpu()
            nct_list = nct[:num_reqs].tolist()
        else:
            seq = common_attn_metadata.seq_lens[:num_reqs]
            if hasattr(seq, "detach"):
                seq = seq.detach()
            if hasattr(seq, "cpu") and getattr(seq, "is_cuda", False):
                seq = seq.cpu()
            seq_list = seq.tolist()
            nct_list = [
                int(seq_list[i]) - int(qsl_list[i + 1] - qsl_list[i])
                for i in range(num_reqs)
            ]

        num_actual = int(
            getattr(common_attn_metadata, "num_actual_tokens", hidden_states.shape[0])
        )
        num_actual = min(num_actual, int(hidden_states.shape[0]))
        hs = hidden_states[:num_actual]
        if hs.numel() == 0:
            return

        _, num_layers, hidden_size = hs.shape
        dtype = hs.dtype
        chunk = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_gather_chunk_tokens", 4096
            )
        )
        if chunk <= 0:
            chunk = num_actual

        if not getattr(self, "_logged_cpu_pages", False):
            logger.info(
                "HS_CPU_PAGES: streaming DtoH to TP0 CPU pages "
                "(chunk=%s tokens, layers=%s, hidden=%s, dtype=%s)",
                chunk,
                num_layers,
                hidden_size,
                dtype,
            )
            self._logged_cpu_pages = True

        copy_stream = self._get_copy_stream()
        ready_event = torch.cuda.Event()
        ready_event.record()
        copy_stream.wait_event(ready_event)

        staging = self._hs_staging
        if (
            staging is None
            or staging.shape[0] < num_actual
            or staging.dtype != dtype
            or staging.shape[1] != num_layers
            or staging.shape[2] != hidden_size
        ):
            try:
                staging = torch.empty(
                    (num_actual, num_layers, hidden_size),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                )
            except Exception:
                staging = torch.empty(
                    (num_actual, num_layers, hidden_size),
                    dtype=dtype,
                    device="cpu",
                )
            self._hs_staging = staging

        with torch.cuda.stream(copy_stream):
            for start in range(0, num_actual, chunk):
                end = min(start + chunk, num_actual)
                staging[start:end].copy_(hs[start:end], non_blocking=True)
                chunk_done = torch.cuda.Event()
                chunk_done.record(copy_stream)
                chunk_done.synchronize()

        for i in range(num_reqs):
            q0 = int(qsl_list[i])
            q1 = int(qsl_list[i + 1])
            q1 = min(q1, num_actual)
            n = q1 - q0
            if n <= 0:
                continue
            req_id = req_ids[i]
            start = max(int(nct_list[i]), 0)
            need = start + n
            cap = need
            if requests is not None and req_id in requests:
                try:
                    cap = max(cap, int(requests[req_id].num_prompt_tokens), need)
                except Exception:
                    cap = need
            page = self._cpu_pages.get(req_id)
            page_is_new = page is None
            if page is None or page.shape[0] < need:
                try:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                if page is not None:
                    new_page[: page.shape[0]].copy_(page)
                self._cpu_pages[req_id] = new_page
                page = new_page
            if page_is_new and start > 0:
                self._fill_prefix_from_hash(req_id, page, start, requests)
            page[start:need].copy_(staging[q0:q1])
            self._publish_full_blocks(req_id, page, start, need, requests)

    @staticmethod
    def _write_tensors(
'''
    text = _replace_once(text, old_write, new_write, "save_hs_batch insert")

    old_submit = """        if not self._is_tp_rank_zero:
            return
        assert self._kv_cache is not None
"""
    new_submit = """        if not self._is_tp_rank_zero:
            self._cpu_pages.pop(pending.req_id, None)
            self._req_block_hashes.pop(pending.req_id, None)
            return
        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: finish is CPU concat + save_file, no NPU gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                self._req_block_hashes.pop(pending.req_id, None)
                return
            page = self._cpu_pages.pop(pending.req_id, None)
            self._req_block_hashes.pop(pending.req_id, None)
            num_tokens = int(pending.token_ids.shape[0])
            if page is None:
                logger.error(
                    "HS_CPU_PAGES: no CPU hidden states for req_id=%s",
                    pending.req_id,
                )
                return
            if page.shape[0] < num_tokens:
                logger.error(
                    "HS_CPU_PAGES: CPU page too short for req_id=%s: %s < %s",
                    pending.req_id,
                    page.shape[0],
                    num_tokens,
                )
                num_tokens = int(page.shape[0])
            hidden_states_cpu = page[:num_tokens].contiguous().clone()
            del page
            copy_done = torch.cuda.Event()
            copy_done.record()
            tensors = {
                "hidden_states": hidden_states_cpu,
                "token_ids": pending.token_ids.clone(),
            }
            os.makedirs(os.path.dirname(pending.filename) or ".", exist_ok=True)
            lock_fd = self._lock_fds.pop(pending.req_id, None)
            if lock_fd is None and self.use_lock:
                lock_path = pending.filename + ".lock"
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            future = self._executor.submit(
                self._write_tensors, tensors, copy_done, pending.filename, lock_fd
            )
            self._req_copy_events[pending.req_id] = copy_done
            self._req_futures[pending.req_id] = future
            future.add_done_callback(partial(self._on_write_done, pending.req_id))
            return
        assert self._kv_cache is not None
"""
    text = _replace_once(text, old_submit, new_submit, "_submit_async_write CPU")

    old_finished = """        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
        )
        return True, {"hidden_states_path": filename}
"""
    new_finished = """        discard = False
        try:
            from vllm.v1.request import RequestStatus

            discard = request.status == RequestStatus.FINISHED_ABORTED
        except Exception:
            discard = False
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
            discard=discard,
        )
        # HS_CPU_PAGES: MLA blocks do not hold HS; free them immediately.
        delay = not getattr(self, "_hs_store_cpu", False)
        params = None if discard else {"hidden_states_path": filename}
        return delay, params
"""
    text = _replace_once(text, old_finished, new_finished, "request_finished")

    old_all_groups = """        return self.request_finished(request, block_ids[self._cache_kv_group_id])
"""
    new_all_groups = """        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: no hidden-state KV group / block ids.
            return self.request_finished(request, [])
        return self.request_finished(request, block_ids[self._cache_kv_group_id])
"""
    return _replace_once(
        text, old_all_groups, new_all_groups, "request_finished_all_groups"
    )


def upgrade_connector_prefix_hash(text: str) -> str:
    """Upgrade an HS_CPU_PAGES connector to add HS_CPU_PREFIX_HASH."""
    old_init_tail = """        self._cpu_pages: dict[str, torch.Tensor] = {}
        self._hs_staging: torch.Tensor | None = None
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)
        if self._hs_store_cpu:
            logger.info(
                "HS_CPU_PAGES: hidden states stored on TP0 CPU, not in NPU KV"
            )
"""
    new_init_tail = """        self._cpu_pages: dict[str, torch.Tensor] = {}
        self._hs_staging: torch.Tensor | None = None
        from collections import OrderedDict
        cc = vllm_config.cache_config
        self._hs_prefix_index_on = bool(
            self._hs_store_cpu and getattr(cc, "enable_prefix_caching", False)
        )
        self._hs_hash_blocks: OrderedDict = OrderedDict()
        self._hs_hash_tokens = 0
        self._hs_prefix_max_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_prefix_cache_max_tokens", 131072
            )
        )
        self._req_block_hashes: dict[str, list] = {}
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)
        if self._hs_store_cpu:
            logger.info(
                "HS_CPU_PAGES: hidden states stored on TP0 CPU, not in NPU KV"
            )
            if self._hs_prefix_index_on:
                logger.info(
                    "HS_CPU_PREFIX_HASH: enabled max_tokens=%s block_size=%s",
                    self._hs_prefix_max_tokens,
                    self._block_size,
                )
            else:
                logger.info(
                    "HS_CPU_PREFIX_HASH: disabled (engine prefix cache off)"
                )
"""
    text = _replace_once(text, old_init_tail, new_init_tail, "upgrade init prefix hash")

    old_save_intro = """    def save_hs_batch(
"""
    new_save_intro = """    def _hs_prefix_enabled(self) -> bool:
        return bool(
            getattr(self, "_hs_store_cpu", False)
            and getattr(self, "_hs_prefix_index_on", False)
            and getattr(self, "_is_tp_rank_zero", False)
        )

    def _block_extra_keys(self, req_state):
        extra = []
        lora = getattr(req_state, "lora_request", None)
        if lora is not None:
            extra.append(getattr(lora, "lora_name", None) or str(lora))
        return tuple(extra) if extra else None

    def _ensure_req_hashes(self, req_id, requests, n_blocks):
        hashes = self._req_block_hashes.setdefault(req_id, [])
        if n_blocks <= 0 or len(hashes) >= n_blocks:
            return hashes[: max(n_blocks, 0)]
        if requests is None or req_id not in requests:
            return hashes
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core import kv_cache_utils as _kvu
        from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

        st = requests[req_id]
        ids = list(st.prompt_token_ids or [])
        ids.extend(st.output_token_ids or [])
        bs = int(self._block_size)
        algo = getattr(
            self._vllm_config.cache_config, "prefix_caching_hash_algo", "sha256"
        )
        hash_fn = get_hash_fn_by_name(algo)
        if getattr(_kvu, "NONE_HASH", None) is None:
            init_none_hash(hash_fn)
        extra = self._block_extra_keys(st)
        while len(hashes) < n_blocks:
            i = len(hashes)
            lo = i * bs
            hi = lo + bs
            if hi > len(ids):
                break
            prev = hashes[-1] if hashes else None
            hashes.append(hash_block_tokens(hash_fn, prev, ids[lo:hi], extra))
        return hashes

    def _evict_hs_hash_blocks(self) -> None:
        max_tok = int(getattr(self, "_hs_prefix_max_tokens", 0) or 0)
        blocks = getattr(self, "_hs_hash_blocks", None)
        if blocks is None or max_tok <= 0:
            return
        while self._hs_hash_tokens > max_tok and blocks:
            _, old = blocks.popitem(last=False)
            self._hs_hash_tokens -= int(old.shape[0])

    def _fill_prefix_from_hash(self, req_id, page, n_prefix, requests) -> int:
        # HS_CPU_PREFIX_HASH: copy cached full blocks into a new request page.
        if not self._hs_prefix_enabled() or n_prefix <= 0:
            return 0
        bs = int(self._block_size)
        n_blocks = n_prefix // bs
        if n_blocks <= 0:
            return 0
        hashes = self._ensure_req_hashes(req_id, requests, n_blocks)
        copied = 0
        for i, h in enumerate(hashes[:n_blocks]):
            blk = self._hs_hash_blocks.get(h)
            if blk is None:
                logger.warning(
                    "HS_CPU_PREFIX_HASH: miss req_id=%s block=%s/%s",
                    req_id,
                    i,
                    n_blocks,
                )
                break
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi or blk.shape[0] != bs:
                break
            page[lo:hi].copy_(blk)
            self._hs_hash_blocks.move_to_end(h)
            copied += 1
        if copied:
            logger.info(
                "HS_CPU_PREFIX_HASH: filled %s/%s prefix blocks req_id=%s",
                copied,
                n_blocks,
                req_id,
            )
        return copied * bs

    def _publish_full_blocks(self, req_id, page, start, need, requests) -> None:
        if not self._hs_prefix_enabled() or need <= start:
            return
        bs = int(self._block_size)
        first = start // bs
        last = need // bs
        if last <= first:
            return
        hashes = self._ensure_req_hashes(req_id, requests, last)
        for i in range(first, min(last, len(hashes))):
            h = hashes[i]
            if h in self._hs_hash_blocks:
                self._hs_hash_blocks.move_to_end(h)
                continue
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi:
                break
            self._hs_hash_blocks[h] = page[lo:hi].contiguous().clone()
            self._hs_hash_tokens += bs
        self._evict_hs_hash_blocks()

    def save_hs_batch(
"""
    text = _replace_once(
        text, old_save_intro, new_save_intro, "upgrade insert prefix helpers"
    )

    old_page_write = """            page = self._cpu_pages.get(req_id)
            if page is None or page.shape[0] < need:
                try:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                if page is not None:
                    new_page[: page.shape[0]].copy_(page)
                self._cpu_pages[req_id] = new_page
                page = new_page
            page[start:need].copy_(staging[q0:q1])
"""
    new_page_write = """            page = self._cpu_pages.get(req_id)
            page_is_new = page is None
            if page is None or page.shape[0] < need:
                try:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                if page is not None:
                    new_page[: page.shape[0]].copy_(page)
                self._cpu_pages[req_id] = new_page
                page = new_page
            if page_is_new and start > 0:
                self._fill_prefix_from_hash(req_id, page, start, requests)
            page[start:need].copy_(staging[q0:q1])
            self._publish_full_blocks(req_id, page, start, need, requests)
"""
    text = _replace_once(
        text, old_page_write, new_page_write, "upgrade save_hs_batch prefix copy"
    )

    old_submit_pops = """        if not self._is_tp_rank_zero:
            self._cpu_pages.pop(pending.req_id, None)
            return
        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: finish is CPU concat + save_file, no NPU gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                return
            page = self._cpu_pages.pop(pending.req_id, None)
"""
    new_submit_pops = """        if not self._is_tp_rank_zero:
            self._cpu_pages.pop(pending.req_id, None)
            self._req_block_hashes.pop(pending.req_id, None)
            return
        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: finish is CPU concat + save_file, no NPU gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                self._req_block_hashes.pop(pending.req_id, None)
                return
            page = self._cpu_pages.pop(pending.req_id, None)
            self._req_block_hashes.pop(pending.req_id, None)
"""
    return _replace_once(
        text, old_submit_pops, new_submit_pops, "upgrade submit pop hashes"
    )


def upgrade_none_hash_init(text: str) -> str:
    old = """        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import hash_block_tokens

        st = requests[req_id]
        ids = list(st.prompt_token_ids or [])
        ids.extend(st.output_token_ids or [])
        bs = int(self._block_size)
        algo = getattr(
            self._vllm_config.cache_config, "prefix_caching_hash_algo", "sha256"
        )
        hash_fn = get_hash_fn_by_name(algo)
        extra = self._block_extra_keys(st)
"""
    new = """        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core import kv_cache_utils as _kvu
        from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

        st = requests[req_id]
        ids = list(st.prompt_token_ids or [])
        ids.extend(st.output_token_ids or [])
        bs = int(self._block_size)
        algo = getattr(
            self._vllm_config.cache_config, "prefix_caching_hash_algo", "sha256"
        )
        hash_fn = get_hash_fn_by_name(algo)
        if getattr(_kvu, "NONE_HASH", None) is None:
            init_none_hash(hash_fn)
        extra = self._block_extra_keys(st)
"""
    return _replace_once(text, old, new, "upgrade init_none_hash")


CONNECTOR_UPSTREAM_METHODS = r'''    def _hs_prefix_enabled(self) -> bool:
        # HS_CPU_UPSTREAM: getattr keeps this safe before register_kv_caches.
        return bool(
            getattr(self, "_hs_store_cpu", False)
            and getattr(self, "_hs_prefix_index_on", False)
            and getattr(self, "_is_tp_rank_zero", False)
        )

    @staticmethod
    def _tensor_on_device(t) -> bool:
        dev = getattr(t, "device", None)
        if dev is not None:
            return str(getattr(dev, "type", "cpu")) != "cpu"
        return bool(getattr(t, "is_cuda", False) or getattr(t, "is_npu", False))

    @staticmethod
    def _host_list(t, n: int) -> list:
        if hasattr(t, "detach"):
            t = t.detach()
        if ExampleHiddenStatesConnector._tensor_on_device(t) and hasattr(t, "cpu"):
            t = t.cpu()
        return t[:n].tolist()

    def _req_hash_scope_ok(self, req_state) -> bool:
        # Token-hash index cannot prove rows that depend on MM / prompt embeds.
        return not (
            getattr(req_state, "mm_features", None)
            or getattr(req_state, "prompt_embeds", None) is not None
            or getattr(req_state, "prompt_is_token_ids", None) is not None
        )

    def _req_state(self, req_id, requests):
        if requests is None:
            return None
        getter = getattr(requests, "get", None)
        if getter is not None:
            return getter(req_id)
        try:
            return requests[req_id]
        except Exception:
            return None

    def _warn_prefix_scope_gap(self, req_id, hit_tokens, reason) -> None:
        logger.warning(
            "HS_CPU_UPSTREAM: req_id=%s skipped backfill of %s prefix-hit "
            "tokens (%s). Those rows stay unfilled; disable prefix caching "
            "if this request's data must be exact.",
            req_id,
            hit_tokens,
            reason,
        )

    def _block_extra_keys(self, req_state):
        extra = []
        lora = getattr(req_state, "lora_request", None)
        if lora is not None:
            extra.append(getattr(lora, "lora_name", None) or str(lora))
        return tuple(extra) if extra else None

    def _ensure_req_hashes(self, req_id, requests, n_blocks):
        hashes = self._req_block_hashes.setdefault(req_id, [])
        if n_blocks <= 0 or len(hashes) >= n_blocks:
            return hashes[: max(n_blocks, 0)]
        st = self._req_state(req_id, requests)
        if st is None:
            return hashes
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core import kv_cache_utils as _kvu
        from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

        ids = list(st.prompt_token_ids or [])
        ids.extend(st.output_token_ids or [])
        bs = int(self._block_size)
        algo = getattr(
            self._vllm_config.cache_config, "prefix_caching_hash_algo", "sha256"
        )
        hash_fn = get_hash_fn_by_name(algo)
        if getattr(_kvu, "NONE_HASH", None) is None:
            init_none_hash(hash_fn)
        extra = self._block_extra_keys(st)
        while len(hashes) < n_blocks:
            i = len(hashes)
            lo = i * bs
            hi = lo + bs
            if hi > len(ids):
                break
            prev = hashes[-1] if hashes else None
            hashes.append(hash_block_tokens(hash_fn, prev, ids[lo:hi], extra))
        return hashes

    def _evict_hs_hash_blocks(self) -> None:
        max_tok = int(getattr(self, "_hs_prefix_max_tokens", 0) or 0)
        blocks = getattr(self, "_hs_hash_blocks", None)
        if blocks is None or max_tok <= 0:
            return
        while self._hs_hash_tokens > max_tok and blocks:
            _, old = blocks.popitem(last=False)
            self._hs_hash_tokens -= int(old.shape[0])

    def _fill_prefix_from_hash(self, req_id, page, n_prefix, requests) -> int:
        if not self._hs_prefix_enabled() or n_prefix <= 0:
            return 0
        st = self._req_state(req_id, requests)
        if st is None:
            self._warn_prefix_scope_gap(
                req_id, n_prefix, "request state unavailable on this rank"
            )
            return 0
        if not self._req_hash_scope_ok(st):
            self._warn_prefix_scope_gap(
                req_id,
                n_prefix,
                "multimodal or prompt-embeds inputs are outside the token-hash scope",
            )
            return 0
        bs = int(self._block_size)
        n_blocks = n_prefix // bs
        if n_blocks <= 0:
            return 0
        hashes = self._ensure_req_hashes(req_id, requests, n_blocks)
        copied = 0
        for i, h in enumerate(hashes[:n_blocks]):
            blk = self._hs_hash_blocks.get(h)
            if blk is None:
                break
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi or blk.shape[0] != bs:
                break
            page[lo:hi].copy_(blk)
            self._hs_hash_blocks.move_to_end(h)
            copied += 1
        if copied < n_blocks:
            logger.warning(
                "HS_CPU_PREFIX_HASH: req_id=%s backfilled %s/%s prefix "
                "blocks; rows from token %s on stay unfilled",
                req_id,
                copied,
                n_blocks,
                copied * bs,
            )
        elif copied:
            logger.info(
                "HS_CPU_PREFIX_HASH: filled %s/%s prefix blocks req_id=%s",
                copied,
                n_blocks,
                req_id,
            )
        return copied * bs

    def _publish_full_blocks(self, req_id, page, start, need, requests) -> None:
        if not self._hs_prefix_enabled() or need <= start:
            return
        st = self._req_state(req_id, requests)
        if st is None or not self._req_hash_scope_ok(st):
            return
        bs = int(self._block_size)
        first = start // bs
        last = need // bs
        if last <= first:
            return
        hashes = self._ensure_req_hashes(req_id, requests, last)
        for i in range(first, min(last, len(hashes))):
            h = hashes[i]
            if h in self._hs_hash_blocks:
                self._hs_hash_blocks.move_to_end(h)
                continue
            lo = i * bs
            hi = lo + bs
            if page.shape[0] < hi:
                break
            if page.is_floating_point() and bool(page[lo:hi].isnan().any()):
                continue
            self._hs_hash_blocks[h] = page[lo:hi].contiguous().clone()
            self._hs_hash_tokens += bs
        self._evict_hs_hash_blocks()

    def save_hs_batch(
        self,
        hidden_states: torch.Tensor,
        common_attn_metadata,
        req_ids: list[str] | None,
        requests=None,
    ) -> None:
        """DtoH this step's hidden states into per-request CPU pages (TP0).

        HS_CPU_UPSTREAM. ``hidden_states`` is [num_tokens, num_layers, hidden].
        Write offset prefers host ``num_computed_tokens``; prefix-hit rows are
        backfilled from the CPU hash index when the request is in token-hash
        scope. Unproven rows keep a NaN sentinel.
        """
        if not getattr(self, "_hs_store_cpu", False):
            return
        if not self._is_tp_rank_zero:
            return
        if req_ids is None or common_attn_metadata is None:
            return
        if hidden_states.numel() == 0:
            return

        num_reqs = int(common_attn_metadata.num_reqs)
        if num_reqs <= 0:
            return

        qsl = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if qsl is None:
            qsl = common_attn_metadata.query_start_loc
        qsl_list = self._host_list(qsl, num_reqs + 1)

        nct_list = self._host_num_computed_tokens(req_ids, requests, num_reqs)
        if nct_list is None:
            seq = common_attn_metadata.seq_lens[:num_reqs]
            seq_list = self._host_list(seq, num_reqs)
            nct_list = [
                int(seq_list[i]) - int(qsl_list[i + 1] - qsl_list[i])
                for i in range(num_reqs)
            ]
        self._save_hs_to_pages(
            hidden_states,
            int(getattr(common_attn_metadata, "num_actual_tokens", hidden_states.shape[0])),
            qsl_list,
            nct_list,
            req_ids,
            requests,
        )

    def save_hs_batch_tokens(
        self,
        hidden_states: torch.Tensor,
        req_ids: list[str] | None,
        query_start_loc_cpu,
        seq_lens,
        num_actual_tokens: int,
        requests=None,
    ) -> None:
        """save_hs_batch for runners without CommonAttentionMetadata (MR V2)."""
        if not getattr(self, "_hs_store_cpu", False):
            return
        if not self._is_tp_rank_zero:
            return
        if req_ids is None or hidden_states.numel() == 0:
            return
        num_reqs = len(req_ids)
        if num_reqs <= 0:
            return
        qsl_list = self._host_list(query_start_loc_cpu, num_reqs + 1)
        nct_list = self._host_num_computed_tokens(req_ids, requests, num_reqs)
        if nct_list is None:
            seq_list = self._host_list(seq_lens, num_reqs)
            nct_list = [
                int(seq_list[i]) - int(qsl_list[i + 1] - qsl_list[i])
                for i in range(num_reqs)
            ]
        self._save_hs_to_pages(
            hidden_states,
            int(num_actual_tokens),
            qsl_list,
            nct_list,
            req_ids,
            requests,
        )

    @staticmethod
    def _host_num_computed_tokens(req_ids, requests, num_reqs):
        if requests is None:
            return None
        nct_list = []
        for i in range(num_reqs):
            st = None
            if i < len(req_ids):
                getter = getattr(requests, "get", None)
                st = getter(req_ids[i]) if getter is not None else None
            nct = getattr(st, "num_computed_tokens", None) if st is not None else None
            if nct is None:
                return None
            nct_list.append(int(nct))
        return nct_list

    def _save_hs_to_pages(
        self,
        hidden_states: torch.Tensor,
        num_actual_tokens: int,
        qsl_list: list,
        nct_list: list,
        req_ids: list,
        requests,
    ) -> None:
        num_reqs = len(req_ids)
        num_actual = min(int(num_actual_tokens), int(hidden_states.shape[0]))
        hs = hidden_states[:num_actual]
        if hs.numel() == 0:
            return

        _, num_layers, hidden_size = hs.shape
        dtype = hs.dtype
        chunk = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_gather_chunk_tokens", 4096
            )
        )
        if chunk <= 0:
            chunk = num_actual

        if not getattr(self, "_logged_cpu_pages", False):
            logger.info(
                "HS_CPU_PAGES: streaming DtoH to TP0 CPU pages "
                "(chunk=%s tokens, layers=%s, hidden=%s, dtype=%s)",
                chunk,
                num_layers,
                hidden_size,
                dtype,
            )
            self._logged_cpu_pages = True

        if self._tensor_on_device(hs):
            copy_stream = self._get_copy_stream()
            ready_event = torch.cuda.Event()
            ready_event.record()
            copy_stream.wait_event(ready_event)
            staging = self._hs_staging
            if (
                staging is None
                or staging.shape[0] < num_actual
                or staging.dtype != dtype
                or staging.shape[1] != num_layers
                or staging.shape[2] != hidden_size
            ):
                try:
                    staging = torch.empty(
                        (num_actual, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    staging = torch.empty(
                        (num_actual, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                self._hs_staging = staging
            with torch.cuda.stream(copy_stream):
                for start in range(0, num_actual, chunk):
                    end = min(start + chunk, num_actual)
                    staging[start:end].copy_(hs[start:end], non_blocking=True)
                    chunk_done = torch.cuda.Event()
                    chunk_done.record(copy_stream)
                    chunk_done.synchronize()
        else:
            staging = hs

        for i in range(num_reqs):
            q0 = int(qsl_list[i])
            q1 = min(int(qsl_list[i + 1]), num_actual)
            n = q1 - q0
            if n <= 0:
                continue
            req_id = req_ids[i]
            start = max(int(nct_list[i]), 0)
            need = start + n
            cap = need
            st = self._req_state(req_id, requests)
            if st is not None:
                try:
                    cap = max(cap, int(st.num_prompt_tokens), need)
                except Exception:
                    cap = need
            page = self._cpu_pages.get(req_id)
            page_is_new = page is None
            if page is None or page.shape[0] < need:
                try:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                except Exception:
                    new_page = torch.empty(
                        (cap, num_layers, hidden_size),
                        dtype=dtype,
                        device="cpu",
                    )
                if new_page.is_floating_point():
                    new_page.fill_(float("nan"))
                if page is not None:
                    new_page[: page.shape[0]].copy_(page)
                self._cpu_pages[req_id] = new_page
                page = new_page
            if page_is_new and start > 0:
                self._fill_prefix_from_hash(req_id, page, start, requests)
            page[start:need].copy_(staging[q0:q1])
            self._publish_full_blocks(req_id, page, start, need, requests)
'''


def upgrade_connector_upstream(text: str) -> str:
    """Bring an HS_CPU_PREFIX_HASH connector up to feat/hs-cpu-pages + mrv2."""
    old_init = """        from collections import OrderedDict
        cc = vllm_config.cache_config
        self._hs_prefix_index_on = bool(
            self._hs_store_cpu and getattr(cc, "enable_prefix_caching", False)
        )
        self._hs_hash_blocks: OrderedDict = OrderedDict()
        self._hs_hash_tokens = 0
        self._hs_prefix_max_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_prefix_cache_max_tokens", 131072
            )
        )
        self._req_block_hashes: dict[str, list] = {}
"""
    new_init = """        from collections import OrderedDict
        # HS_CPU_UPSTREAM: latch the prefix index in register_kv_caches
        # (after model load may flip enable_prefix_caching).
        self._logged_cpu_pages = False
        self._hs_prefix_index_on = False
        self._hs_hash_blocks: OrderedDict = OrderedDict()
        self._hs_hash_tokens = 0
        self._hs_prefix_max_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "hs_prefix_cache_max_tokens", 131072
            )
        )
        self._req_block_hashes: dict[str, list] = {}
"""
    text = _replace_once(text, old_init, new_init, "upstream init latch prefix")

    old_init_log = """            if self._hs_prefix_index_on:
                logger.info(
                    "HS_CPU_PREFIX_HASH: enabled max_tokens=%s block_size=%s",
                    self._hs_prefix_max_tokens,
                    self._block_size,
                )
            else:
                logger.info(
                    "HS_CPU_PREFIX_HASH: disabled (engine prefix cache off)"
                )
"""
    new_init_log = """            logger.info(
                "HS_CPU_UPSTREAM: prefix index resolved in register_kv_caches"
            )
"""
    if old_init_log in text:
        text = _replace_once(text, old_init_log, new_init_log, "upstream init log")

    old_reg = """            logger.info(
                "HS_CPU_PAGES: register_kv_caches skipped NPU HS group (tp0=%s)",
                self._is_tp_rank_zero,
            )
            return
"""
    new_reg = """            self._hs_prefix_index_on = bool(
                self._vllm_config.cache_config.enable_prefix_caching
            ) and (self._hs_prefix_max_tokens > 0)
            if self._hs_prefix_index_on:
                logger.info(
                    "HS_CPU_PREFIX_HASH: enabled max_tokens=%s block_size=%s",
                    self._hs_prefix_max_tokens,
                    self._block_size,
                )
            else:
                logger.info(
                    "HS_CPU_PREFIX_HASH: disabled (engine prefix cache off "
                    "or max_tokens=0)"
                )
            logger.info(
                "HS_CPU_PAGES: register_kv_caches skipped NPU HS group (tp0=%s)",
                self._is_tp_rank_zero,
            )
            return
"""
    text = _replace_once(text, old_reg, new_reg, "upstream register prefix latch")

    if "def get_finished_count(" not in text:
        old_find = """    @classmethod
    def _find_cache_kv_group_id(cls, kv_cache_config: "KVCacheConfig | None") -> int:
"""
        new_find = """    def get_finished_count(self) -> int | None:
        # HS_CPU_UPSTREAM: only TP rank 0 writes, so aggregator expects 1.
        return 1

    @classmethod
    def _find_cache_kv_group_id(cls, kv_cache_config: "KVCacheConfig | None") -> int:
"""
        text = _replace_once(text, old_find, new_find, "get_finished_count")

    text = _replace_span(
        text,
        "    def _hs_prefix_enabled(self) -> bool:",
        "            self._publish_full_blocks(req_id, page, start, need, requests)\n",
        CONNECTOR_UPSTREAM_METHODS.rstrip() + "\n",
        "upstream helpers+save_hs_batch",
    )

    old_write = """    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
    ) -> None:
        \"\"\"Thread worker: wait for async DtoH copy, write to disk, release lock.

        ``lock_fd`` is an open file descriptor on the companion ``.lock``
        file with ``LOCK_EX`` already held.  Closing it releases the lock,
        which unblocks any client sleeping on ``LOCK_SH``.
        \"\"\"
        try:
            event.synchronize()
            save_file(tensors, filename)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)  # releases LOCK_EX
"""
    new_write = '''    @staticmethod
    def _write_tensors(
        tensors: dict[str, torch.Tensor],
        event: torch.cuda.Event,
        filename: str,
        lock_fd: int | None,
        req_id: str | None = None,
        audit_nan: bool = False,
    ) -> None:
        """Thread worker: wait for async DtoH copy, write to disk, release lock.

        ``lock_fd`` is an open file descriptor on the companion ``.lock``
        file with ``LOCK_EX`` already held.  Closing it releases the lock,
        which unblocks any client sleeping on ``LOCK_SH``.

        HS_CPU_UPSTREAM: with ``audit_nan`` count rows still holding the NaN
        sentinel before writing so holes are loud instead of looking valid.
        """
        try:
            event.synchronize()
            if audit_nan:
                hs = tensors["hidden_states"]
                if hs.is_floating_point():
                    hole_rows = int(
                        torch.isnan(hs)
                        .any(dim=tuple(range(1, hs.dim())))
                        .sum()
                    )
                    if hole_rows:
                        logger.warning(
                            "req_id=%s: %s/%s hidden-state rows were never "
                            "filled (NaN sentinel); prefix-cache rows missed "
                            "by the CPU index or a dropped streaming step",
                            req_id,
                            hole_rows,
                            hs.shape[0],
                        )
            save_file(tensors, filename)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)  # releases LOCK_EX
'''
    text = _replace_once(text, old_write, new_write, "upstream _write_tensors audit")

    old_cpu_submit = """        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: finish is CPU concat + save_file, no NPU gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                self._req_block_hashes.pop(pending.req_id, None)
                return
            page = self._cpu_pages.pop(pending.req_id, None)
            self._req_block_hashes.pop(pending.req_id, None)
            num_tokens = int(pending.token_ids.shape[0])
            if page is None:
                logger.error(
                    "HS_CPU_PAGES: no CPU hidden states for req_id=%s",
                    pending.req_id,
                )
                return
            if page.shape[0] < num_tokens:
                logger.error(
                    "HS_CPU_PAGES: CPU page too short for req_id=%s: %s < %s",
                    pending.req_id,
                    page.shape[0],
                    num_tokens,
                )
                num_tokens = int(page.shape[0])
            hidden_states_cpu = page[:num_tokens].contiguous().clone()
            del page
            copy_done = torch.cuda.Event()
            copy_done.record()
            tensors = {
                "hidden_states": hidden_states_cpu,
                "token_ids": pending.token_ids.clone(),
            }
            os.makedirs(os.path.dirname(pending.filename) or ".", exist_ok=True)
            lock_fd = self._lock_fds.pop(pending.req_id, None)
            if lock_fd is None and self.use_lock:
                lock_path = pending.filename + ".lock"
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            future = self._executor.submit(
                self._write_tensors, tensors, copy_done, pending.filename, lock_fd
            )
            self._req_copy_events[pending.req_id] = copy_done
            self._req_futures[pending.req_id] = future
            future.add_done_callback(partial(self._on_write_done, pending.req_id))
            return
"""
    new_cpu_submit = """        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_UPSTREAM: finish is CPU concat + save_file, no device gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                self._req_block_hashes.pop(pending.req_id, None)
                return
            finalized = self._finalize_cpu_page(pending.req_id, pending.token_ids)
            if finalized is None:
                return
            hidden_states_cpu, token_ids = finalized
            copy_done = torch.cuda.Event()
            copy_done.record()
            tensors = {
                "hidden_states": hidden_states_cpu,
                "token_ids": token_ids,
            }
            os.makedirs(os.path.dirname(pending.filename) or ".", exist_ok=True)
            lock_fd = self._lock_fds.pop(pending.req_id, None)
            if lock_fd is None and self.use_lock:
                lock_path = pending.filename + ".lock"
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            future = self._executor.submit(
                self._write_tensors,
                tensors,
                copy_done,
                pending.filename,
                lock_fd,
                pending.req_id,
                True,
            )
            self._req_copy_events[pending.req_id] = copy_done
            self._req_futures[pending.req_id] = future
            future.add_done_callback(partial(self._on_write_done, pending.req_id))
            return
"""
    text = _replace_once(text, old_cpu_submit, new_cpu_submit, "upstream finalize submit")

    if "def _finalize_cpu_page(" not in text:
        old_sched = """    # ==============================
    # Scheduler-side methods
    # ==============================
"""
        new_sched = '''    def _finalize_cpu_page(self, req_id: str, token_ids: torch.Tensor):
        """Pop a finished request's CPU page and align it with its token ids."""
        page = self._cpu_pages.pop(req_id, None)
        self._req_block_hashes.pop(req_id, None)
        if page is None:
            logger.error("HS_CPU_PAGES: no CPU hidden states for req_id=%s", req_id)
            return None
        num_tokens = int(token_ids.shape[0])
        if page.shape[0] < num_tokens:
            logger.error(
                "HS_CPU_PAGES: CPU page too short for req_id=%s: %s < %s; "
                "saving the first %s tokens only",
                req_id,
                page.shape[0],
                num_tokens,
                page.shape[0],
            )
            num_tokens = int(page.shape[0])
            token_ids = token_ids[:num_tokens]
        return page[:num_tokens].contiguous(), token_ids

    # ==============================
    # Scheduler-side methods
    # ==============================
'''
        if old_sched in text:
            text = _replace_once(text, old_sched, new_sched, "upstream _finalize_cpu_page")
        else:
            # Insert before get_num_new_matched_tokens if the banner is missing.
            old_gn = """    def get_num_new_matched_tokens(
"""
            text = _replace_once(
                text,
                old_gn,
                new_sched + "    def get_num_new_matched_tokens(\n",
                "upstream _finalize_cpu_page via get_num",
            )

    old_fn = """        req_id = request.request_id
        filename = self._request_filenames.pop(req_id)
"""
    new_fn = """        req_id = request.request_id
        filename = self._request_filenames.pop(req_id, None)
        if filename is None:
            self._pending_saves.pop(req_id, None)
            return False, None
"""
    text = _replace_once(text, old_fn, new_fn, "upstream request_finished missing file")

    old_poll = """        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            event = self._req_copy_events.get(req_id)
            if event is None or event.query():
                self._req_copy_events.pop(req_id, None)
                done_sending.add(req_id)
                self._accumulated_finished_req_ids.discard(req_id)
                # Clean up any leftover lock fds (e.g. aborted requests
                # that never went through _submit_async_write).
                lock_fd = self._lock_fds.pop(req_id, None)
                if lock_fd is not None:
                    os.close(lock_fd)

        return done_sending or None, None
"""
    new_poll = """        done_sending: set[str] = set()
        for req_id in list(self._accumulated_finished_req_ids):
            event = self._req_copy_events.get(req_id)
            if event is not None and not event.query():
                continue
            self._req_copy_events.pop(req_id, None)
            self._accumulated_finished_req_ids.discard(req_id)
            if event is not None:
                done_sending.add(req_id)
            lock_fd = self._lock_fds.pop(req_id, None)
            if lock_fd is not None:
                os.close(lock_fd)

        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_UPSTREAM: delay_free=False already dropped the req_id;
            # reporting finished_sending would trip the scheduler assertion.
            return None, None
        return done_sending or None, None
"""
    text = _replace_once(text, old_poll, new_poll, "upstream get_finished silent")

    old_all = """        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: no hidden-state KV group / block ids.
            return self.request_finished(request, [])
        return self.request_finished(request, block_ids[self._cache_kv_group_id])
"""
    new_all = """        if not 0 <= self._cache_kv_group_id < len(block_ids):
            return False, None
        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: no hidden-state KV group / block ids.
            return self.request_finished(request, [])
        return self.request_finished(request, block_ids[self._cache_kv_group_id])
"""
    text = _replace_once(text, old_all, new_all, "upstream all_groups bounds")
    return text


def upgrade_proposer_upstream(text: str) -> str:
    """Replace silent CPU skip with the loud-fail V1 stream path."""
    if "requires an active KV transfer" in text:
        return text
    old = """            if has_kv_transfer_group() and common_attn_metadata is not None:
                connector = get_kv_transfer_group()
                save_fn = getattr(connector, "save_hs_batch", None)
                if save_fn is not None:
                    runner = getattr(self, "runner", None)
                    req_ids = None
                    requests = None
                    if runner is not None:
                        n_req = int(common_attn_metadata.num_reqs)
                        req_ids = list(runner.input_batch.req_ids[:n_req])
                        requests = getattr(runner, "requests", None)
                    save_fn(
                        self.hidden_states[:num_tokens],
                        common_attn_metadata,
                        req_ids,
                        requests,
                    )
            return sampled_token_ids[:, :1]
"""
    new = """            if not has_kv_transfer_group():
                raise ValueError(
                    "hs_store_device=cpu requires an active KV transfer "
                    "group to stream hidden states to CPU pages"
                )
            connector = get_kv_transfer_group()
            save_fn = getattr(connector, "save_hs_batch", None)
            if save_fn is None:
                raise ValueError(
                    "hs_store_device=cpu requires a KV connector that "
                    "implements save_hs_batch (e.g. "
                    f"ExampleHiddenStatesConnector), got "
                    f"{type(connector).__name__}"
                )
            runner = getattr(self, "runner", None)
            if runner is None:
                raise ValueError(
                    "hs_store_device=cpu requires the drafter to hold a "
                    "model runner reference to resolve per-request ids"
                )
            if common_attn_metadata is None:
                raise ValueError(
                    "hs_store_device=cpu requires common_attn_metadata"
                )
            n_req = int(common_attn_metadata.num_reqs)
            save_fn(
                self.hidden_states[:num_tokens],
                common_attn_metadata,
                list(runner.input_batch.req_ids[:n_req]),
                getattr(runner, "requests", None),
            )
            return sampled_token_ids[:, :1]
"""
    text = _replace_once(text, old, new, "proposer loud fail")
    if "def __init__(self, vllm_config: VllmConfig, device, runner=None):" not in text:
        old_init = """    def __init__(self, vllm_config: VllmConfig, device):
        assert vllm_config.speculative_config is not None
"""
        new_init = """    def __init__(self, vllm_config: VllmConfig, device, runner=None):
        assert vllm_config.speculative_config is not None
        # HS_CPU_UPSTREAM: keep a runner set by an Ascend subclass before super().
        self.runner = runner if runner is not None else getattr(self, "runner", None)
"""
        if old_init in text:
            text = _replace_once(text, old_init, new_init, "proposer init runner upgrade")
    return text


def patch_kv_transfer(text: str) -> str:
    if "def is_hs_store_cpu(" in text:
        return text
    old = """    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)
"""
    new = """    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)

    def is_hs_store_cpu(self) -> bool:
        # HS_CPU_UPSTREAM: single source of truth for hs_store_device.
        return (
            str(self.get_from_extra_config("hs_store_device", "device")).lower()
            == "cpu"
        )
"""
    text = _replace_once(text, old, new, "KVTransferConfig.is_hs_store_cpu")
    if "\ndef is_hs_store_cpu(" not in text:
        text = text.rstrip() + """


def is_hs_store_cpu(vllm_config) -> bool:
    # HS_CPU_UPSTREAM
    kvt = getattr(vllm_config, "kv_transfer_config", None)
    return kvt is not None and kvt.is_hs_store_cpu()
"""
        if not text.endswith("\n"):
            text += "\n"
    return text


def patch_gpu_model_runner(text: str) -> str:
    if UPSTREAM_MARKER in text:
        return text
    if "from vllm.model_executor.models.extract_hidden_states import" not in text:
        old_imp = """from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
"""
        if old_imp in text:
            text = _replace_once(
                text,
                old_imp,
                old_imp
                + "from vllm.model_executor.models.extract_hidden_states import (\n"
                "    CacheOnlyAttentionLayer,\n"
                ")\n",
                "gpu_model_runner CacheOnly import",
            )
    old_drafter = """                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
"""
    new_drafter = """                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device, runner=self
                )
"""
    if old_drafter in text:
        text = _replace_once(text, old_drafter, new_drafter, "gpu_model_runner runner=")
    old_spec = """            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
"""
    new_spec = """            # HS_CPU_UPSTREAM: keep CacheOnlyAttentionLayer out of the device pool.
            if isinstance(attn_module, CacheOnlyAttentionLayer):
                _kvt = self.vllm_config.kv_transfer_config
                _hs_store = "device"
                if _kvt is not None:
                    _hs_store = str(
                        _kvt.get_from_extra_config("hs_store_device", "device")
                    ).lower()
                if _hs_store == "cpu":
                    continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
"""
    if "CacheOnlyAttentionLayer" in text and old_spec in text and "HS_CPU_UPSTREAM: keep CacheOnly" not in text:
        text = _replace_once(text, old_spec, new_spec, "gpu_model_runner skip CacheOnly")
    return text


def patch_attn_utils(text: str) -> str:
    if UPSTREAM_MARKER in text:
        return text
    old = """    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
"""
    new = """    from vllm.model_executor.models.extract_hidden_states import (
        CacheOnlyAttentionLayer,
    )

    for layer_name, attn_module in attn_layers.items():
        # HS_CPU_UPSTREAM: skip the cache-only layer when HS streams to CPU.
        _kvt = vllm_config.kv_transfer_config
        if _kvt is not None and isinstance(attn_module, CacheOnlyAttentionLayer):
            if str(_kvt.get_from_extra_config("hs_store_device", "device")).lower() == "cpu":
                continue
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
"""
    return _replace_once(text, old, new, "attn_utils skip CacheOnly")


def patch_v2_speculator(text: str) -> str:
    if UPSTREAM_MARKER in text:
        return text
    old_imp = """from typing import Any

import torch
import torch.nn as nn
"""
    new_imp = """from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
"""
    if "class _HsReqScope" not in text:
        text = _replace_once(text, old_imp, new_imp, "v2 dataclass import")
        old_cls = """class ExtractHiddenStatesSpeculator(DraftModelSpeculator):
"""
        new_cls = '''@dataclass
class _HsReqScope:
    """Per-request inputs the hidden-states CPU prefix index needs (MR V2)."""

    num_prompt_tokens: int
    mm_features: Any = None
    prompt_embeds: Any = None
    prompt_is_token_ids: Any = None
    lora_request: Any = None
    prompt_token_ids: Any = None
    output_token_ids: Any = None
    num_computed_tokens: int = 0


class ExtractHiddenStatesSpeculator(DraftModelSpeculator):
'''
        text = _replace_once(text, old_cls, new_cls, "v2 _HsReqScope")

    old_init_end = """        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_hidden_states,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.dtype,
            device=device,
        )
"""
    new_init_end = """        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_hidden_states,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.dtype,
            device=device,
        )
        kvt = vllm_config.kv_transfer_config
        store = "cpu"
        if kvt is not None:
            store = str(kvt.get_from_extra_config("hs_store_device", "cpu")).lower()
        self._hs_store_cpu = store == "cpu"
        self._hs_scope: dict[str, _HsReqScope] = {}
        self.req_states = None
"""
    text = _replace_once(text, old_init_end, new_init_end, "v2 speculator cpu flag")

    old_propose_del = """        del (
            last_hidden_states,
            num_sampled,
            num_rejected,
            next_prefill_tokens,
            temperature,
            seeds,
            dummy_run,
            mm_inputs,
            is_profile,
        )

        draft_tokens = last_sampled[input_batch.idx_mapping, :1]
        if skip_attn_for_dummy_run:
            return draft_tokens
"""
    new_propose_del = """        del (
            last_hidden_states,
            num_sampled,
            num_rejected,
            next_prefill_tokens,
            temperature,
            seeds,
            mm_inputs,
        )

        draft_tokens = last_sampled[input_batch.idx_mapping, :1]
        if skip_attn_for_dummy_run:
            return draft_tokens
"""
    text = _replace_once(text, old_propose_del, new_propose_del, "v2 propose keep dummy/profile")

    old_copy = """        stacked_hidden_states = torch.stack(aux_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]
        self.hidden_states[:num_tokens].copy_(stacked_hidden_states)

        draft_attn_metadata = {
"""
    new_copy = """        stacked_hidden_states = torch.stack(aux_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]
        self.hidden_states[:num_tokens].copy_(stacked_hidden_states)

        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_UPSTREAM: stream this step and skip the cache-only forward.
            if not dummy_run:
                self._refresh_hs_scope_tokens(input_batch)
                self._stream_hidden_states_to_cpu(
                    input_batch, num_tokens, is_profile
                )
            return draft_tokens

        draft_attn_metadata = {
"""
    text = _replace_once(text, old_copy, new_copy, "v2 propose cpu stream")

    # Append helper methods before the last blank of the class by inserting
    # after load_model's last raise block.
    old_load_end = """            raise ValueError(
                "ExtractHiddenStatesModel should have exactly one attention "
                f"layer, found {len(self.draft_attn_layer_names)}"
            )
"""
    new_load_end = '''            raise ValueError(
                "ExtractHiddenStatesModel should have exactly one attention "
                f"layer, found {len(self.draft_attn_layer_names)}"
            )

    def on_new_request(self, req_id: str, new_req_data) -> None:
        if not getattr(self, "_hs_store_cpu", False):
            return
        prompt_ids = list(
            getattr(new_req_data, "prefill_token_ids", None)
            or getattr(new_req_data, "prompt_token_ids", None)
            or []
        )
        self._hs_scope[req_id] = _HsReqScope(
            num_prompt_tokens=int(getattr(new_req_data, "prompt_len", 0) or 0),
            mm_features=getattr(new_req_data, "mm_features", None),
            prompt_embeds=getattr(new_req_data, "prompt_embeds", None),
            prompt_is_token_ids=getattr(new_req_data, "prompt_is_token_ids", None),
            lora_request=getattr(new_req_data, "lora_request", None),
            prompt_token_ids=prompt_ids,
            output_token_ids=[],
            num_computed_tokens=int(
                getattr(new_req_data, "num_computed_tokens", 0) or 0
            ),
        )

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        for req_id in finished_req_ids:
            self._hs_scope.pop(req_id, None)

    def _refresh_hs_scope_tokens(self, input_batch) -> None:
        if self.req_states is None:
            return
        for req_id in input_batch.req_ids[: input_batch.num_reqs]:
            scope = self._hs_scope.get(req_id)
            if scope is None:
                continue
            row = self.req_states.req_id_to_index.get(req_id)
            if row is None:
                continue
            nct = getattr(self.req_states, "num_computed_tokens_np", None)
            if nct is not None:
                scope.num_computed_tokens = int(nct[row])
            total_t = getattr(self.req_states, "total_len", None)
            ids_t = getattr(self.req_states, "all_token_ids", None)
            total = None
            if total_t is not None:
                gpu = getattr(total_t, "gpu", None)
                np_ = getattr(total_t, "np", None)
                if gpu is not None:
                    total = int(gpu[row].item())
                elif np_ is not None:
                    total = int(np_[row])
            if ids_t is not None and total is not None:
                gpu = getattr(ids_t, "gpu", None)
                np_ = getattr(ids_t, "np", None)
                if gpu is not None:
                    scope.prompt_token_ids = gpu[row, :total].tolist()
                elif np_ is not None:
                    scope.prompt_token_ids = np_[row, :total].tolist()
                scope.output_token_ids = []

    def _stream_hidden_states_to_cpu(
        self, input_batch, num_tokens: int, is_profile: bool = False
    ) -> None:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            if is_profile:
                return
            raise ValueError(
                "hs_store_device=cpu with extract_hidden_states V2 has no "
                "active KV transfer group, so hidden states would be lost"
            )
        connector = get_kv_transfer_group()
        save_fn = getattr(connector, "save_hs_batch_tokens", None)
        if save_fn is None:
            raise ValueError(
                "hs_store_device=cpu requires save_hs_batch_tokens on the "
                f"KV connector, got {type(connector).__name__}"
            )
        num_reqs = input_batch.num_reqs
        save_fn(
            self.hidden_states[:num_tokens],
            list(input_batch.req_ids[:num_reqs]),
            input_batch.query_start_loc_np[: num_reqs + 1],
            input_batch.seq_lens[:num_reqs],
            num_tokens,
            requests=self._hs_scope,
        )
'''
    return _replace_once(text, old_load_end, new_load_end, "v2 speculator helpers")


def patch_v2_base_speculator(text: str) -> str:
    if UPSTREAM_MARKER in text:
        return text
    if "def on_new_request(" in text:
        return text
    old = """        is_profile: bool = False,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
"""
    new = """        is_profile: bool = False,
    ) -> torch.Tensor:
        pass

    def on_new_request(self, req_id: str, new_req_data) -> None:
        # HS_CPU_UPSTREAM: no-op; ExtractHiddenStatesSpeculator overrides.
        return

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        return


class DraftModelSpeculator(BaseSpeculator):
"""
    return _replace_once(text, old, new, "BaseSpeculator HS hooks")


def patch_v2_init_speculator(text: str) -> str:
    if UPSTREAM_MARKER in text:
        return text
    old = """def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
"""
    new = """def init_speculator(vllm_config: VllmConfig, device: torch.device, req_states=None):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        spec = ExtractHiddenStatesSpeculator(vllm_config, device)
        if req_states is not None:
            spec.req_states = req_states  # HS_CPU_UPSTREAM
        return spec
"""
    return _replace_once(text, old, new, "init_speculator req_states")


def patch_v2_model_runner(text: str) -> str:
    if UPSTREAM_MARKER in text:
        return text
    old_rs = """        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
        )
"""
    new_rs = """        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        if self.speculator is not None:
            # HS_CPU_UPSTREAM: speculator is built before RequestState.
            self.speculator.req_states = self.req_states
"""
    if "self.speculator.req_states = self.req_states" not in text:
        text = _replace_once(text, old_rs, new_rs, "v2 inject req_states")

    old_add = """            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
"""
    new_add = """            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            if self.speculator is not None:
                on_new = getattr(self.speculator, "on_new_request", None)
                if on_new is not None:
                    on_new(req_id, new_req_data)
"""
    if "on_new_request" not in text or "on_new(req_id, new_req_data)" not in text:
        text = _replace_once(text, old_add, new_add, "v2 on_new_request")

    old_fin = """        for req_id in sorted(finished_req_ids):
            self._remove_request(req_id)
"""
    new_fin = """        if self.speculator is not None:
            on_fin = getattr(self.speculator, "on_requests_finished", None)
            if on_fin is not None:
                on_fin(set(scheduler_output.finished_req_ids))
        for req_id in sorted(finished_req_ids):
            self._remove_request(req_id)
"""
    if "on_requests_finished" not in text or "on_fin(set(scheduler_output.finished_req_ids))" not in text:
        text = _replace_once(text, old_fin, new_fin, "v2 on_requests_finished")
    return text


CONNECTOR_BLOCK_TABLE_METHODS = r'''    def _infer_num_gpu_blocks(self, kv_caches=None) -> int:
        # HS_CPU_BLOCK_TABLE: same capacity as the NPU hybrid BlockPool.
        # Ascend initialize_from_config does not write cache_config.num_gpu_blocks.
        kvc = getattr(self, "_kv_cache_config", None)
        n = getattr(kvc, "num_blocks", None) if kvc is not None else None
        if n:
            return int(n)
        vcfg = getattr(self, "_vllm_config", None)
        cc = getattr(vcfg, "cache_config", None) if vcfg is not None else None
        n = getattr(cc, "num_gpu_blocks", None) if cc is not None else None
        if n:
            return int(n)
        tok = getattr(cc, "kv_cache_size_tokens", None) if cc is not None else None
        bs = int(getattr(self, "_block_size", 0) or 0)
        if tok and bs:
            return int(tok) // bs
        vals = []
        if isinstance(kv_caches, dict):
            vals = kv_caches.values()
        elif kv_caches is not None:
            vals = kv_caches

        def _shape_of(t):
            if t is None:
                return None
            if isinstance(t, (list, tuple)):
                for x in t:
                    s = _shape_of(x)
                    if s:
                        return s
                return None
            if hasattr(t, "shape"):
                return tuple(int(x) for x in t.shape)
            return None

        for t in vals:
            shape = _shape_of(t)
            if not shape:
                continue
            if len(shape) >= 2 and shape[0] in (1, 2) and shape[1] > 8:
                return int(shape[1])
            return int(shape[0])
        return 0

    @staticmethod
    def _flatten_block_ids(raw):
        if raw is None:
            return None
        if len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        try:
            return [int(x) for x in list(raw)]
        except (TypeError, ValueError):
            return None

    def _ensure_cpu_hs_table(self, num_layers, hidden_size, dtype) -> None:
        table = getattr(self, "_cpu_hs_blocks", None)
        nblocks = int(getattr(self, "_cpu_hs_num_blocks", 0) or 0)
        if nblocks <= 0:
            nblocks = self._infer_num_gpu_blocks()
            self._cpu_hs_num_blocks = nblocks
        bs = int(self._block_size)
        if nblocks <= 0 or bs <= 0:
            logger.warning(
                "HS_CPU_BLOCK_TABLE: cannot allocate table num_blocks=%s block_size=%s",
                nblocks,
                bs,
            )
            return
        if (
            table is not None
            and table.shape[0] == nblocks
            and table.shape[1] == bs
            and table.shape[2] == num_layers
            and table.shape[3] == hidden_size
            and table.dtype == dtype
        ):
            return
        table = torch.empty(
            (nblocks, bs, num_layers, hidden_size),
            dtype=dtype,
            device="cpu",
        )
        if table.is_floating_point():
            table.fill_(float("nan"))
        self._cpu_hs_blocks = table
        logger.info(
            "HS_CPU_BLOCK_TABLE: allocated [%s, %s, %s, %s] dtype=%s (~%.2f GiB)",
            nblocks,
            bs,
            num_layers,
            hidden_size,
            dtype,
            table.numel() * table.element_size() / (1024 ** 3),
        )

    def _req_block_ids(self, req_id, requests):
        st = self._req_state(req_id, requests) if requests is not None else None
        bids = None
        if st is not None:
            bids = self._flatten_block_ids(getattr(st, "block_ids", None))
        if bids:
            return bids
        cache = getattr(self, "_req_block_ids_cache", None)
        if cache is not None:
            return cache.get(req_id)
        return None

    def _copy_from_block_table(self, dest, bids, n_tokens) -> int:
        table = getattr(self, "_cpu_hs_blocks", None)
        if table is None or not bids or n_tokens <= 0:
            return 0
        bs = int(self._block_size)
        copied = 0
        t = 0
        n_tokens = min(int(n_tokens), int(dest.shape[0]))
        while t < n_tokens:
            bidx = t // bs
            if bidx >= len(bids):
                break
            bid = int(bids[bidx])
            off = t % bs
            n = min(bs - off, n_tokens - t)
            if bid < 0 or bid >= table.shape[0]:
                t += n
                continue
            src = table[bid, off : off + n]
            if (
                dest.is_floating_point()
                and src.is_floating_point()
                and bool(src.isnan().any())
            ):
                t += n
                continue
            dest[t : t + n].copy_(src)
            copied += n
            t += n
        return copied

    def _fill_prefix_from_block_ids(self, req_id, page, n_prefix, requests) -> int:
        if not self._hs_prefix_enabled() or n_prefix <= 0:
            return 0
        table = getattr(self, "_cpu_hs_blocks", None)
        if table is None:
            logger.warning(
                "HS_CPU_BLOCK_TABLE: no CPU block table; cannot backfill req_id=%s",
                req_id,
            )
            return 0
        bids = self._req_block_ids(req_id, requests)
        if not bids:
            logger.warning(
                "HS_CPU_BLOCK_TABLE: no block_ids for prefix req_id=%s",
                req_id,
            )
            return 0
        copied = self._copy_from_block_table(page, bids, n_prefix)
        if copied < n_prefix:
            logger.warning(
                "HS_CPU_BLOCK_TABLE: req_id=%s gathered %s/%s prefix tokens",
                req_id,
                copied,
                n_prefix,
            )
        elif copied:
            logger.info(
                "HS_CPU_BLOCK_TABLE: gathered %s prefix tokens req_id=%s",
                copied,
                req_id,
            )
        return copied

    def _scatter_hs_to_blocks(self, req_id, page, start, need, requests) -> None:
        table = getattr(self, "_cpu_hs_blocks", None)
        if table is None:
            return
        bids = self._req_block_ids(req_id, requests)
        if not bids:
            logger.warning(
                "HS_CPU_BLOCK_TABLE: no block_ids to scatter req_id=%s",
                req_id,
            )
            return
        bs = int(self._block_size)
        t = int(start)
        need = min(int(need), int(page.shape[0]))
        while t < need:
            bidx = t // bs
            if bidx >= len(bids):
                break
            bid = int(bids[bidx])
            off = t % bs
            n = min(bs - off, need - t)
            if 0 <= bid < table.shape[0]:
                table[bid, off : off + n].copy_(page[t : t + n])
            t += n

    def _materialize_cpu_page_from_blocks(self, req_id, num_tokens) -> None:
        """Assemble dump rows from cpu_hs[block_id] before the id can be reused."""
        table = getattr(self, "_cpu_hs_blocks", None)
        bids = getattr(self, "_req_block_ids_cache", {}).get(req_id)
        if table is None or not bids or num_tokens <= 0:
            return
        gathered = torch.empty(
            (int(num_tokens), table.shape[2], table.shape[3]),
            dtype=table.dtype,
            device="cpu",
        )
        if gathered.is_floating_point():
            gathered.fill_(float("nan"))
        self._copy_from_block_table(gathered, bids, num_tokens)
        self._cpu_pages[req_id] = gathered

    def _cpu_page_has_holes(self, req_id, token_ids) -> bool:
        page = self._cpu_pages.get(req_id)
        if page is None:
            return True
        n = int(token_ids.shape[0])
        if n <= 0 or int(page.shape[0]) < n:
            return True
        sl = page[:n]
        return bool(sl.is_floating_point() and torch.isnan(sl).any())

'''


def upgrade_connector_block_table(text: str) -> str:
    """Address Host HS by the same BlockPool block_ids as NPU MLA KV."""
    if BLOCK_TABLE_MARKER in text:
        return text

    old_init = """        self._logged_cpu_pages = False
        self._hs_prefix_index_on = False
"""
    new_init = """        self._logged_cpu_pages = False
        self._hs_prefix_index_on = False
        self._cpu_hs_blocks = None
        self._cpu_hs_num_blocks = 0
        self._req_block_ids_cache: dict[str, list] = {}
        # HS_CPU_BLOCK_TABLE
"""
    text = _replace_once(text, old_init, new_init, "block-table init fields")

    old_reg = """            self._hs_prefix_index_on = bool(
                self._vllm_config.cache_config.enable_prefix_caching
            ) and (self._hs_prefix_max_tokens > 0)
            if self._hs_prefix_index_on:
                logger.info(
                    "HS_CPU_PREFIX_HASH: enabled max_tokens=%s block_size=%s",
                    self._hs_prefix_max_tokens,
                    self._block_size,
                )
            else:
                logger.info(
                    "HS_CPU_PREFIX_HASH: disabled (engine prefix cache off "
                    "or max_tokens=0)"
                )
            logger.info(
                "HS_CPU_PAGES: register_kv_caches skipped NPU HS group (tp0=%s)",
                self._is_tp_rank_zero,
            )
            return
"""
    new_reg = """            self._hs_prefix_index_on = bool(
                self._vllm_config.cache_config.enable_prefix_caching
            )
            self._cpu_hs_num_blocks = self._infer_num_gpu_blocks(kv_caches)
            self._cpu_hs_blocks = None
            logger.info(
                "HS_CPU_BLOCK_TABLE: num_blocks=%s block_size=%s prefix=%s tp0=%s",
                self._cpu_hs_num_blocks,
                self._block_size,
                self._hs_prefix_index_on,
                self._is_tp_rank_zero,
            )
            logger.info(
                "HS_CPU_PAGES: register_kv_caches skipped NPU HS group (tp0=%s)",
                self._is_tp_rank_zero,
            )
            return
"""
    text = _replace_once(text, old_reg, new_reg, "block-table register_kv_caches")

    old_req_state = """    def _req_state(self, req_id, requests):
        if requests is None:
            return None
        getter = getattr(requests, "get", None)
        if getter is not None:
            return getter(req_id)
        try:
            return requests[req_id]
        except Exception:
            return None

    def _warn_prefix_scope_gap(self, req_id, hit_tokens, reason) -> None:
"""
    new_req_state = (
        """    def _req_state(self, req_id, requests):
        if requests is None:
            return None
        getter = getattr(requests, "get", None)
        if getter is not None:
            return getter(req_id)
        try:
            return requests[req_id]
        except Exception:
            return None

"""
        + CONNECTOR_BLOCK_TABLE_METHODS
        + "    def _warn_prefix_scope_gap(self, req_id, hit_tokens, reason) -> None:\n"
    )
    text = _replace_once(
        text, old_req_state, new_req_state, "block-table helper methods"
    )

    old_save = """            if page_is_new and start > 0:
                self._fill_prefix_from_hash(req_id, page, start, requests)
            page[start:need].copy_(staging[q0:q1])
            self._publish_full_blocks(req_id, page, start, need, requests)
"""
    new_save = """            self._ensure_cpu_hs_table(num_layers, hidden_size, dtype)
            bids = self._req_block_ids(req_id, requests)
            if bids:
                self._req_block_ids_cache[req_id] = bids
            if page_is_new and start > 0:
                self._fill_prefix_from_block_ids(req_id, page, start, requests)
            page[start:need].copy_(staging[q0:q1])
            self._scatter_hs_to_blocks(req_id, page, start, need, requests)
"""
    text = _replace_once(text, old_save, new_save, "block-table scatter/gather save")

    old_fin = """    def _finalize_cpu_page(self, req_id: str, token_ids: torch.Tensor):
        \"\"\"Pop a finished request's CPU page and align it with its token ids.\"\"\"
        page = self._cpu_pages.pop(req_id, None)
        self._req_block_hashes.pop(req_id, None)
        if page is None:
            logger.error("HS_CPU_PAGES: no CPU hidden states for req_id=%s", req_id)
            return None
        num_tokens = int(token_ids.shape[0])
        if page.shape[0] < num_tokens:
            logger.error(
                "HS_CPU_PAGES: CPU page too short for req_id=%s: %s < %s; "
                "saving the first %s tokens only",
                req_id,
                page.shape[0],
                num_tokens,
                page.shape[0],
            )
            num_tokens = int(page.shape[0])
            token_ids = token_ids[:num_tokens]
        return page[:num_tokens].contiguous(), token_ids
"""
    new_fin = """    def _finalize_cpu_page(self, req_id: str, token_ids: torch.Tensor):
        \"\"\"Pop a finished request's CPU page; refuse dump if any row is NaN.\"\"\"
        # HS_CPU_WORKER_REFUSE: this runs on the Worker, which owns _cpu_pages.
        self._req_block_hashes.pop(req_id, None)
        bids = self._req_block_ids_cache.pop(req_id, None)
        page = self._cpu_pages.pop(req_id, None)
        if page is None:
            logger.error("HS_CPU_PAGES: no CPU hidden states for req_id=%s", req_id)
            return None
        num_tokens = int(token_ids.shape[0])
        if page.shape[0] < num_tokens:
            logger.error(
                "HS_CPU_BLOCK_TABLE: CPU page too short for req_id=%s: %s < %s; "
                "refusing dump",
                req_id,
                page.shape[0],
                num_tokens,
            )
            return None
        sl = page[:num_tokens]
        if bids:
            self._copy_from_block_table(sl, bids, num_tokens)
        if sl.is_floating_point() and bool(torch.isnan(sl).any()):
            n_nan = int(torch.isnan(sl.reshape(num_tokens, -1)[:, 0]).sum())
            logger.error(
                "HS_CPU_BLOCK_TABLE: refusing dump for req_id=%s "
                "(%s/%s rows never filled)",
                req_id,
                n_nan,
                num_tokens,
            )
            return None
        return sl.contiguous(), token_ids[:num_tokens]
"""
    text = _replace_once(text, old_fin, new_fin, "block-table refuse-NaN finalize")

    old_finished = """        discard = False
        try:
            from vllm.v1.request import RequestStatus

            discard = request.status == RequestStatus.FINISHED_ABORTED
        except Exception:
            discard = False
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
            discard=discard,
        )
        # HS_CPU_PAGES: MLA blocks do not hold HS; free them immediately.
        delay = not getattr(self, "_hs_store_cpu", False)
        params = None if discard else {"hidden_states_path": filename}
        return delay, params
"""
    new_finished = """        discard = False
        try:
            from vllm.v1.request import RequestStatus

            discard = request.status == RequestStatus.FINISHED_ABORTED
        except Exception:
            discard = False
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
            discard=discard,
        )
        # HS_CPU_WORKER_REFUSE: hole check is Worker _finalize_cpu_page.
        delay = not getattr(self, "_hs_store_cpu", False)
        params = None if discard else {"hidden_states_path": filename}
        return delay, params
"""
    text = _replace_once(
        text, old_finished, new_finished, "block-table request_finished path only"
    )

    old_all = """        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: no hidden-state KV group / block ids.
            return self.request_finished(request, [])
"""
    new_all = """        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_BLOCK_TABLE: hybrid MLA ids (group 0) address CPU HS slots.
            mla_ids = []
            if block_ids:
                mla_ids = (
                    block_ids[0]
                    if isinstance(block_ids[0], (list, tuple))
                    else block_ids
                )
            return self.request_finished(request, mla_ids)
"""
    text = _replace_once(
        text, old_all, new_all, "block-table request_finished_all_groups"
    )
    return text


_WORKER_REFUSE_FINISHED_OLD = """        discard = False
        try:
            from vllm.v1.request import RequestStatus

            discard = request.status == RequestStatus.FINISHED_ABORTED
        except Exception:
            discard = False
        incomplete = False
        if not discard and getattr(self, "_hs_store_cpu", False):
            bids = getattr(self, "_req_block_ids_cache", {}).get(req_id)
            if not bids:
                bids = self._flatten_block_ids(getattr(request, "block_ids", None))
            if not bids:
                bids = self._flatten_block_ids(block_ids)
            if bids:
                self._req_block_ids_cache[req_id] = bids
            self._materialize_cpu_page_from_blocks(req_id, int(token_ids.shape[0]))
            if self._cpu_page_has_holes(req_id, token_ids):
                logger.error(
                    "HS_CPU_BLOCK_TABLE: refusing dump for req_id=%s "
                    "(NaN holes); client should retry with cache_salt",
                    req_id,
                )
                self._cpu_pages.pop(req_id, None)
                self._req_block_ids_cache.pop(req_id, None)
                discard = True
                incomplete = True
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
            discard=discard,
        )
        # HS_CPU_PAGES: MLA blocks do not hold HS; free them immediately.
        delay = not getattr(self, "_hs_store_cpu", False)
        if incomplete:
            params = {"hidden_states_incomplete": True}
        elif discard:
            params = None
        else:
            params = {"hidden_states_path": filename}
        return delay, params
"""

_WORKER_REFUSE_FINISHED_NEW = """        discard = False
        try:
            from vllm.v1.request import RequestStatus

            discard = request.status == RequestStatus.FINISHED_ABORTED
        except Exception:
            discard = False
        self._pending_saves[req_id] = PendingSave(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids,
            block_ids=list(block_ids),
            discard=discard,
        )
        # HS_CPU_WORKER_REFUSE: hole check is Worker _finalize_cpu_page.
        delay = not getattr(self, "_hs_store_cpu", False)
        params = None if discard else {"hidden_states_path": filename}
        return delay, params
"""

_WORKER_REFUSE_FINALIZE_OLD = """    def _finalize_cpu_page(self, req_id: str, token_ids: torch.Tensor):
        \"\"\"Pop a finished request's CPU page; refuse dump if any row is NaN.\"\"\"
        # HS_CPU_BLOCK_TABLE
        self._req_block_hashes.pop(req_id, None)
        self._req_block_ids_cache.pop(req_id, None)
        page = self._cpu_pages.pop(req_id, None)
        if page is None:
            logger.error("HS_CPU_PAGES: no CPU hidden states for req_id=%s", req_id)
            return None
        num_tokens = int(token_ids.shape[0])
        if page.shape[0] < num_tokens:
            logger.error(
                "HS_CPU_BLOCK_TABLE: CPU page too short for req_id=%s: %s < %s; "
                "refusing dump",
                req_id,
                page.shape[0],
                num_tokens,
            )
            return None
        sl = page[:num_tokens]
        if sl.is_floating_point() and bool(torch.isnan(sl).any()):
            n_nan = int(torch.isnan(sl.reshape(num_tokens, -1)[:, 0]).sum())
            logger.error(
                "HS_CPU_BLOCK_TABLE: refusing dump for req_id=%s "
                "(%s/%s rows never filled)",
                req_id,
                n_nan,
                num_tokens,
            )
            return None
        return sl.contiguous(), token_ids[:num_tokens]
"""

_WORKER_REFUSE_FINALIZE_NEW = """    def _finalize_cpu_page(self, req_id: str, token_ids: torch.Tensor):
        \"\"\"Pop a finished request's CPU page; refuse dump if any row is NaN.\"\"\"
        # HS_CPU_WORKER_REFUSE: this runs on the Worker, which owns _cpu_pages.
        self._req_block_hashes.pop(req_id, None)
        bids = self._req_block_ids_cache.pop(req_id, None)
        page = self._cpu_pages.pop(req_id, None)
        if page is None:
            logger.error("HS_CPU_PAGES: no CPU hidden states for req_id=%s", req_id)
            return None
        num_tokens = int(token_ids.shape[0])
        if page.shape[0] < num_tokens:
            logger.error(
                "HS_CPU_BLOCK_TABLE: CPU page too short for req_id=%s: %s < %s; "
                "refusing dump",
                req_id,
                page.shape[0],
                num_tokens,
            )
            return None
        sl = page[:num_tokens]
        if bids:
            self._copy_from_block_table(sl, bids, num_tokens)
        if sl.is_floating_point() and bool(torch.isnan(sl).any()):
            n_nan = int(torch.isnan(sl.reshape(num_tokens, -1)[:, 0]).sum())
            logger.error(
                "HS_CPU_BLOCK_TABLE: refusing dump for req_id=%s "
                "(%s/%s rows never filled)",
                req_id,
                n_nan,
                num_tokens,
            )
            return None
        return sl.contiguous(), token_ids[:num_tokens]
"""

_WORKER_REFUSE_SUBMIT_OLD = """            finalized = self._finalize_cpu_page(pending.req_id, pending.token_ids)
            if finalized is None:
                return
"""

_WORKER_REFUSE_SUBMIT_NEW = """            finalized = self._finalize_cpu_page(pending.req_id, pending.token_ids)
            if finalized is None:
                # HS_CPU_WORKER_REFUSE: scheduler already returned the path;
                # tell the client not to wait 600s for a file that will never appear.
                try:
                    marker = pending.filename + ".incomplete"
                    os.makedirs(os.path.dirname(pending.filename) or ".", exist_ok=True)
                    with open(marker, "w"):
                        pass
                except Exception:
                    logger.exception(
                        "HS_CPU_WORKER_REFUSE: failed to write incomplete marker "
                        "for req_id=%s",
                        pending.req_id,
                    )
                lock_fd = self._lock_fds.pop(pending.req_id, None)
                if lock_fd is not None:
                    os.close(lock_fd)
                try:
                    os.remove(pending.filename + ".lock")
                except OSError:
                    pass
                return
"""


def upgrade_connector_worker_refuse(text: str) -> str:
    """Move NaN refuse off EngineCore request_finished onto Worker wait_for_save."""
    if WORKER_REFUSE_MARKER in text and pending_incomplete_marker_written(text):
        return text
    if _WORKER_REFUSE_FINISHED_OLD in text:
        text = _replace_once(
            text,
            _WORKER_REFUSE_FINISHED_OLD,
            _WORKER_REFUSE_FINISHED_NEW,
            "worker-refuse request_finished",
        )
    elif "self._materialize_cpu_page_from_blocks(req_id, int(token_ids.shape[0]))" in text:
        raise ValueError("worker-refuse: EngineCore materialize span not found")
    if _WORKER_REFUSE_FINALIZE_OLD in text:
        text = _replace_once(
            text,
            _WORKER_REFUSE_FINALIZE_OLD,
            _WORKER_REFUSE_FINALIZE_NEW,
            "worker-refuse finalize fill-from-table",
        )
    if _WORKER_REFUSE_SUBMIT_OLD in text:
        text = _replace_once(
            text,
            _WORKER_REFUSE_SUBMIT_OLD,
            _WORKER_REFUSE_SUBMIT_NEW,
            "worker-refuse incomplete marker",
        )
    elif "pending.filename + \".incomplete\"" not in text:
        raise ValueError("worker-refuse: finalize-None submit span not found")
    return text


def pending_incomplete_marker_written(text: str) -> bool:
    return 'pending.filename + ".incomplete"' in text


def upgrade_connector_kv_config_blocks(text: str) -> str:
    """Read BlockPool size from KVCacheConfig.num_blocks (Ascend never sets num_gpu_blocks)."""
    if 'getattr(kvc, "num_blocks"' in text:
        return text
    old_infer = """    def _infer_num_gpu_blocks(self, kv_caches=None) -> int:
        # HS_CPU_BLOCK_TABLE: same capacity as the NPU hybrid BlockPool.
        vcfg = getattr(self, "_vllm_config", None)
        cc = getattr(vcfg, "cache_config", None) if vcfg is not None else None
        n = getattr(cc, "num_gpu_blocks", None) if cc is not None else None
        if n:
            return int(n)
        vals = []
        if isinstance(kv_caches, dict):
            vals = kv_caches.values()
        elif kv_caches is not None:
            vals = kv_caches
        for t in vals:
            if t is None or not hasattr(t, "shape"):
                continue
            shape = tuple(int(x) for x in t.shape)
            if not shape:
                continue
            if len(shape) >= 2 and shape[0] in (1, 2) and shape[1] > 8:
                return int(shape[1])
            return int(shape[0])
        return 0
"""
    # Reuse the methods blob's infer implementation (first method in the blob).
    infer_start = CONNECTOR_BLOCK_TABLE_METHODS.find("    def _infer_num_gpu_blocks")
    infer_end = CONNECTOR_BLOCK_TABLE_METHODS.find("    @staticmethod\n    def _flatten_block_ids")
    new_infer = CONNECTOR_BLOCK_TABLE_METHODS[infer_start:infer_end]
    text = _replace_once(text, old_infer, new_infer, "infer num_blocks from kv_cache_config")

    old_ens = """        nblocks = int(getattr(self, "_cpu_hs_num_blocks", 0) or 0)
        bs = int(self._block_size)
        if nblocks <= 0 or bs <= 0:
            return
"""
    new_ens = """        nblocks = int(getattr(self, "_cpu_hs_num_blocks", 0) or 0)
        if nblocks <= 0:
            nblocks = self._infer_num_gpu_blocks()
            self._cpu_hs_num_blocks = nblocks
        bs = int(self._block_size)
        if nblocks <= 0 or bs <= 0:
            logger.warning(
                "HS_CPU_BLOCK_TABLE: cannot allocate table num_blocks=%s block_size=%s",
                nblocks,
                bs,
            )
            return
"""
    text = _replace_once(text, old_ens, new_ens, "lazy allocate cpu hs table")
    return text


def _patch_connector_full(text: str) -> str:
    if PREFIX_MARKER not in text:
        if MARKER not in text:
            text = patch_connector(text)
        if PREFIX_MARKER not in text and MARKER in text:
            text = upgrade_connector_prefix_hash(text)
        if PREFIX_MARKER in text and "init_none_hash" not in text:
            text = upgrade_none_hash_init(text)
    if UPSTREAM_MARKER not in text or "save_hs_batch_tokens" not in text:
        text = upgrade_connector_upstream(text)
    if BLOCK_TABLE_MARKER not in text:
        text = upgrade_connector_block_table(text)
    if 'getattr(kvc, "num_blocks"' not in text:
        text = upgrade_connector_kv_config_blocks(text)
    if WORKER_REFUSE_MARKER not in text or not pending_incomplete_marker_written(text):
        text = upgrade_connector_worker_refuse(text)
    return text


def _patch_proposer_full(text: str) -> str:
    if MARKER not in text:
        text = patch_proposer(text)
    else:
        text = upgrade_proposer_upstream(text)
    return text


PATCHERS = {
    "model_runner": patch_model_runner,
    "cache_only": patch_cache_only,
    "proposer": _patch_proposer_full,
    "connector": _patch_connector_full,
    "kv_transfer": patch_kv_transfer,
    "gpu_model_runner": patch_gpu_model_runner,
    "attn_utils": patch_attn_utils,
    "v2_speculator": patch_v2_speculator,
    "v2_base_speculator": patch_v2_base_speculator,
    "v2_init_speculator": patch_v2_init_speculator,
    "v2_model_runner": patch_v2_model_runner,
}


def _already_complete(key: str, text: str) -> bool:
    if key == "connector":
        return (
            BLOCK_TABLE_MARKER in text
            and 'getattr(kvc, "num_blocks"' in text
            and WORKER_REFUSE_MARKER in text
            and pending_incomplete_marker_written(text)
        )
    if key == "proposer":
        return "requires an active KV transfer" in text
    if key in OPTIONAL_KEYS or key in ("model_runner", "cache_only"):
        return UPSTREAM_MARKER in text or (
            key in ("model_runner", "cache_only") and MARKER in text
        )
    return MARKER in text and UPSTREAM_MARKER in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=str(default))
    args = parser.parse_args()

    failed = False
    for key, patcher in PATCHERS.items():
        path = Path(getattr(args, key))
        if not path.is_file():
            if key in OPTIONAL_KEYS:
                print(f"skip optional missing: {path}")
                continue
            print(f"{key}: not found: {path}", file=sys.stderr)
            failed = True
            continue
        text = path.read_text()
        if _already_complete(key, text):
            print(f"already patched: {path}")
            continue
        try:
            new_text = patcher(text)
        except ValueError as exc:
            print(f"{key}: {exc}", file=sys.stderr)
            failed = True
            continue
        if new_text == text:
            print(f"unchanged: {path}")
            continue
        if args.dry_run:
            print(f"dry-run: would patch {path}")
            continue
        bak = _backup(path)
        path.write_text(new_text)
        print(f"patched {path}")
        print(f"backup  {bak}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
