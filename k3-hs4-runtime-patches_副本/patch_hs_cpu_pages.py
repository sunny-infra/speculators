#!/usr/bin/env python3
"""Move extract_hidden_states off the NPU hybrid KV pool.

Hidden states stay bf16 and keep all Eagle aux layers. They are copied
DtoH on TP0 each step into per-request CPU pages; request_finished writes
safetensors from those pages. MLA/indexer KV stays on NPU.

Safe to re-run: each target file is skipped when the HS_CPU_PAGES marker
is already present.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

MARKER = "HS_CPU_PAGES"

SITE = Path("/usr/local/python3.11.15/lib/python3.11/site-packages")

DEFAULTS = {
    "model_runner": SITE / "vllm_ascend/worker/model_runner_v1.py",
    "cache_only": SITE / "vllm/model_executor/models/extract_hidden_states.py",
    "proposer": SITE / "vllm/v1/spec_decode/extract_hidden_states.py",
    "connector": SITE
    / "vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py",
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
    old_validate = '''        raise ValueError(f"Cache-only layer {layer!r} not in any KV cache group")
'''
    new_validate = '''        # HS_CPU_PAGES: layer is omitted from NPU groups; reuse group 0
        # common_attn_metadata (query_start_loc / num_computed_tokens).
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
    new_propose = """        # Copy hidden states to buffer
        self.hidden_states[:num_tokens] = stacked_hidden_states

        # HS_CPU_PAGES: stream this step to TP0 CPU pages and skip the
        # cache-only NPU write (there is no HiddenStateCacheSpec group).
        kvt = self.vllm_config.kv_transfer_config
        store = "cpu"
        if kvt is not None:
            store = str(kvt.get_from_extra_config("hs_store_device", "cpu")).lower()
        if store == "cpu":
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
            )

            if has_kv_transfer_group() and common_attn_metadata is not None:
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

        assert self.attn_metadata_builder is not None
"""
    return _replace_once(text, old_propose, new_propose, "proposer.propose CPU DtoH")


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
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)
        if self._hs_store_cpu:
            logger.info(
                "HS_CPU_PAGES: hidden states stored on TP0 CPU, not in NPU KV"
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
    new_write = '''    def save_hs_batch(
        self,
        hidden_states: torch.Tensor,
        common_attn_metadata,
        req_ids: list[str] | None,
        requests=None,
    ) -> None:
        """DtoH this step's hidden states into per-request CPU pages (TP0).

        HS_CPU_PAGES. ``hidden_states`` is [num_tokens, num_layers, hidden].
        Logical write offset is ``num_computed_tokens`` for each request so
        chunked prefill concatenates; prefix-cache hits leave earlier rows.
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
            return
        if getattr(self, "_hs_store_cpu", False):
            # HS_CPU_PAGES: finish is CPU concat + save_file, no NPU gather.
            if pending.discard:
                self._cpu_pages.pop(pending.req_id, None)
                return
            page = self._cpu_pages.pop(pending.req_id, None)
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


PATCHERS = {
    "model_runner": patch_model_runner,
    "cache_only": patch_cache_only,
    "proposer": patch_proposer,
    "connector": patch_connector,
}


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
            print(f"{key}: not found: {path}", file=sys.stderr)
            failed = True
            continue
        text = path.read_text()
        if MARKER in text:
            print(f"already patched: {path}")
            continue
        try:
            new_text = patcher(text)
        except ValueError as exc:
            print(f"{key}: {exc}", file=sys.stderr)
            failed = True
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
