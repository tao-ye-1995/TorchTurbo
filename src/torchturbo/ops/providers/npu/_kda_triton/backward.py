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

"""Packed KDA backward: positive even H, K=V=128, C=64; TND dO and NTD caches."""
from array import array
from functools import lru_cache
import os
import torch
from ._kernels.prepare import _prepare_launcher
from ._finalize import kda_bwd_finalize
from ._kernels._infra import get_npu_properties, get_l2_bypass_offset
from ._kernels._state_launch import kda_bwd_state, select_state_task_shape
_HOST_METADATA_CACHE = {}

@lru_cache(None)
def _k1_config(cann, device):
    return (get_npu_properties()['num_aicore'], get_l2_bypass_offset(device))

@lru_cache(maxsize=32)
def _k2_task_shape(cu, num_aicore, heads=4):
    counts = tuple(((right - left + 63) // 64 for left, right in zip(cu, cu[1:])))
    return select_state_task_shape(sum((count > 0 for count in counts)), num_aicore, counts, heads)

def _host_metadata(cu, indices):
    if isinstance(cu, torch.Tensor):
        cu = cu.detach().cpu().reshape(-1).tolist()
    cu = tuple(cu)
    key = cu
    cached = _HOST_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    if any((left == right for left, right in zip(cu, cu[1:]))):
        cu = (cu[0], *(right for left, right in zip(cu, cu[1:]) if right > left))
        if len(cu) == 1:
            cu = (0, 0)
        indices = None
    offsets = [0]
    for left, right in zip(cu, cu[1:]):
        offsets.append(offsets[-1] + (right - left + 63) // 64)
    if indices is None:
        indices = tuple((value for seq in range(len(cu) - 1) for chunk in range(offsets[seq + 1] - offsets[seq]) for value in (seq, chunk)))
    else:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().reshape(-1).tolist()
        indices = tuple(indices)
    cu_bytes = array("i", cu).tobytes()
    idx_bytes = array("i", indices).tobytes()
    idx_start = (len(cu_bytes) + 15) // 16 * 16
    off_start = idx_start + (len(idx_bytes) + 15) // 16 * 16
    payload = bytearray(cu_bytes + bytes(idx_start - len(cu_bytes)) + idx_bytes + bytes(off_start - idx_start - len(idx_bytes)) + array("q", offsets).tobytes())
    host = torch.frombuffer(payload, dtype=torch.uint8)
    cached = (cu, indices, tuple(offsets), len(cu) - 1 if cu[-1] else 0, host, idx_start, off_start)
    if len(_HOST_METADATA_CACHE) >= 32:
        _HOST_METADATA_CACHE.clear()
    _HOST_METADATA_CACHE[key] = cached
    return cached

def _device_metadata(cu, indices, device):
    cu, indices, offsets, active, host, idx_start, off_start = _host_metadata(cu, indices)
    packed = host.to(device=device)
    cu_dev = packed[:len(cu) * 4].view(torch.int32)
    idx_dev = packed[idx_start:idx_start + len(indices) * 4].view(torch.int32)
    off_dev = packed[off_start:].view(torch.int64)
    return cu, indices, cu_dev, idx_dev, off_dev, active


def npu_chunk_kda_bwd_triton(q, k, v, beta_eff, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale, *, raw_g=None, A_log=None, dt_bias=None, initial_state=None, dht=None, cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True, disable_recompute=True, use_exp2=True, state_v_first=False, implementation='auto', q_rstd=None, k_rstd=None, layout='NTD', use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, prepared_metadata=None):
    """Inputs: q/k/v/w/qg/kg/v_new BF16 [H,T,128]; Aqk/Akk BF16 [H,T,64]; h BF16
    [NC,H,128,128]; gk FP32 [H,T,128]; d_o BF16 [T,H,128]; raw_g NTD [H,T,128]
    FP32; beta_eff NTD [H,T], FP32 with fused sigmoid. A_log/dt_bias FP32;
    q_rstd/k_rstd paired FP32 [H,T]. Outputs:
    (dq,dk,dv,dbeta,dg,None,dA_log,dbias), NTD token gradients; dbeta/dg match
    beta_eff/raw_g dtype. initial_state/dht are unsupported; dt_bias=None yields
    dbias=None. Inputs must satisfy the documented layout/dtype/configuration contract.
    """
    heads, total = v.shape[:2]
    with torch.npu.device(q.device):
        if total == 0:
            dq, dk, dv = (torch.zeros_like(v) for _ in range(3))
            dbias = None if dt_bias is None else torch.zeros_like(dt_bias)
            return (dq, dk, dv, torch.zeros_like(beta_eff), torch.zeros_like(raw_g), None, torch.zeros_like(A_log), dbias)
        cu, indices, cu_dev, idx_dev, off_dev, active = prepared_metadata if prepared_metadata is not None else _device_metadata(cu_seqlens, chunk_indices, q.device)
        da = Aqk.new_empty(heads, total, 64, dtype=torch.float32)
        dv_local = v_new.new_empty(heads, total, 128)
        dqr = Aqk.new_empty(heads, total, 128, dtype=torch.float32)
        cores, offset = _k1_config(os.environ['ASCEND_HOME_PATH'], q.device.index)
        _prepare_launcher[cores,](v_new=v_new, A=Aqk, do=d_o, h=h, dv=dv_local, dA=da, dq_raw=dqr, cu_seqlens=cu_dev, chunk_indices=idx_dev, scale=scale, T_total=total, task_num=(len(indices) // 2) * (heads // 2), num_core=cores, HV=heads, V=128, K=128, BT=64, STATE_V_FIRST=False, CACHE_OFFSET=offset, DO_NTD=False, num_warps=1, num_stages=2)
        dh, _, scan = kda_bwd_state(qg, kg, w, gk, d_o, dv_local, cu_dev, off_dev, len(indices) // 2, scale, do_layout='TND', active_sequences=active, inplace_dv=True, head_window=_k2_task_shape(cu, cores, heads)[0])
        bias = torch.zeros(heads, 128, dtype=torch.float32, device=q.device) if dt_bias is None else dt_bias.reshape(heads, 128)
        dq, dk, dv, db, dg, da_log, dbias = kda_bwd_finalize(
            q, k, v, gk, raw_g, beta_eff, A_log.reshape(heads), bias,
            Akk, v_new, h, dh, scan, da, dqr, cu_dev, idx_dev,
            scale=scale, q_rstd=q_rstd, k_rstd=k_rstd,
            n_chunks=len(indices)//2,
        )
        if dt_bias is not None:
            dbias = dbias.reshape(dt_bias.shape)
        else:
            dbias = None
        return (dq, dk, dv, db, dg, None, da_log, dbias)
