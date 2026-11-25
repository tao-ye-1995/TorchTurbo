# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""K2 host interface for packed NTD tensors and TND/NTD dO."""
import torch
import triton

from ._infra import get_l2_bypass_offset
from ._state_common import COMPILER_OPTIONS
from .state_gradient import _kda_bwd_state_kernel


def _max_head_chunk_load(chunk_counts, heads, num_aicore, total_heads=4):
    """Host chunk counts and integer head/core counts; returns an integer load estimate."""
    groups = total_heads // heads
    cores = min(num_aicore, sum(count > 0 for count in chunk_counts) * groups)
    if cores == 0:
        return 0
    loads = [0] * cores
    for seq, count in enumerate(chunk_counts):
        for group in range(groups):
            loads[(seq * groups + group) % cores] += count * heads
    return max(loads)


def select_state_task_shape(num_sequences, num_aicore, chunk_counts=None, total_heads=4):
    """Return a compatible (head window, value block) for positive even H."""
    if total_heads % 4 == 0 and (total_heads // 2) * num_sequences > num_aicore:
        if (total_heads // 4) * num_sequences > num_aicore and chunk_counts is not None:
            load4 = _max_head_chunk_load(chunk_counts, 4, num_aicore, total_heads)
            load2 = _max_head_chunk_load(chunk_counts, 2, num_aicore, total_heads)
            if load2 * 100 < load4 * 95:
                return 2, 128
        return 4, 128
    if total_heads * num_sequences > num_aicore:
        return 2, 128
    return 1, 128


def _resolve_head_window(total_heads, window):
    """Return a 1/2/4 head window, capped at two for non-four-multiple H."""
    return min(window, 2) if total_heads % 4 else window


def kda_bwd_state(q, k, w, gk, do, dv, cu_seqlens,
        chunk_offsets, n_chunks, scale, *, do_layout='TND', active_sequences=None,
        inplace_dv=False, head_window=None, vblock=128, chunk_counts=None):
    """q/k/w/dv BF16 NTD [H,T,128], gk FP32 NTD, do BF16 TND/NTD. cu_seqlens int32 [N+1],
    chunk_offsets int64 [N+1], n_chunks host integer. Returns (dh,None,dv_scan), with dh
    BF16 [NC,H,128,128] and dv_scan BF16 [H,T,128]. inplace_dv=True returns dv storage
    as dv_scan.
    """
    total_heads = q.shape[0]
    with torch.npu.device(q.device):
        dh = q.new_empty(n_chunks, total_heads, 128, 128)
        scan = dv if inplace_dv else torch.empty_like(dv)
        if n_chunks == 0:
            return dh, None, scan
        sequence_slots = cu_seqlens.numel() - 1
        active = sequence_slots if active_sequences is None else active_sequences
        available = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device())['num_aicore']
        heads = select_state_task_shape(active, available, chunk_counts, total_heads)[0] if head_window is None else head_window
        heads = _resolve_head_window(total_heads, heads)
        parts = (total_heads // heads) * (128 // vblock)

        tasks = sequence_slots * parts
        cores = min(available, active * parts)
        slots = cores * heads
        qb = q.new_empty(slots, 128, vblock)
        wb = torch.empty_like(qb)
        kwargs = dict(q=q, k=k, w=w, gk=gk, do=do, dv=dv, dh=dh, dv2=scan,
            cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
            q_ws=qb, w_ws=wb, scale=scale, T_total=q.shape[1],
            task_num=tasks, num_core=cores, HEADS=total_heads,
            CACHE_OFFSET=get_l2_bypass_offset(q.device.index),
            DO_NTD=do_layout == 'NTD', WINDOW=heads, VBLOCK=vblock,
            INPLACE_DV=inplace_dv, **COMPILER_OPTIONS)
        _kda_bwd_state_kernel[(cores,)](**kwargs)
    return dh, None, scan
