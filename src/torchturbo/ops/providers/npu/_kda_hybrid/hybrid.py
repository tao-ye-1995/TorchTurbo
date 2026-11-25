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
from functools import lru_cache
import os
import torch
from ._kernels.prepare import _prepare_launcher
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
    if len(cu) < 2 or cu[0] != 0 or any((b < a for a, b in zip(cu, cu[1:]))):
        raise ValueError('cu_seqlens must start at 0 and be nondecreasing')
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
    if len(indices) != 2 * offsets[-1]:
        raise ValueError('chunk_indices length must match cu_seqlens')
    cached = (cu, indices, tuple(offsets), sum((right > left for left, right in zip(cu, cu[1:]))))
    if len(_HOST_METADATA_CACHE) >= 32:
        _HOST_METADATA_CACHE.clear()
    _HOST_METADATA_CACHE[key] = cached
    return cached

def _device_metadata(cu, indices, device):
    cu, indices, offsets, active = _host_metadata(cu, indices)
    return (cu, indices, torch.tensor(cu, dtype=torch.int32, device=device), torch.tensor(indices, dtype=torch.int32, device=device), torch.tensor(offsets, dtype=torch.int64, device=device), active)


def _runtime_shape(v, d_o):
    """Check runtime head grouping and the incoming token-major gradient boundary."""
    if v.ndim != 3:
        raise ValueError('hybrid requires packed NTD inputs')
    heads, total, width = v.shape
    if heads <= 0 or heads % 2 or width != 128:
        raise ValueError('hybrid requires positive even H=HV, K=V=128')
    if tuple(d_o.shape) != (total, heads, 128):
        raise ValueError('d_o must have token-major shape [T,H,V]')
    return heads, total

def npu_chunk_kda_bwd_hybrid(q, k, v, beta_eff, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale, *, raw_g=None, A_log=None, dt_bias=None, initial_state=None, dht=None, cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True, disable_recompute=True, use_exp2=True, state_v_first=False, implementation='auto', q_rstd=None, k_rstd=None, layout='NTD', use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, prepared_metadata=None):
    """Inputs: q/k/v/w/qg/kg/v_new BF16 [H,T,128]; Aqk/Akk BF16 [H,T,64]; h BF16
    [NC,H,128,128]; gk FP32 [H,T,128]; d_o BF16 [T,H,128]; raw_g NTD [H,T,128]
    BF16/FP32; beta_eff NTD [H,T], FP32 with fused sigmoid. A_log/dt_bias FP32;
    q_rstd/k_rstd optional paired FP32 [H,T]. Outputs:
    (dq,dk,dv,dbeta,dg,None,dA_log,dbias), NTD token gradients; dbeta/dg match
    beta_eff/raw_g dtype. initial_state/dht are unsupported; dt_bias=None yields
    dbias=None.
    """
    if layout != 'NTD' or v.ndim != 3 or (cu_seqlens is None and prepared_metadata is None):
        raise ValueError('hybrid requires packed NTD inputs/output and cu_seqlens')
    if chunk_size != 64 or not safe_gate or (not use_gate_in_kernel) or (not disable_recompute) or (not use_exp2) or state_v_first:
        raise ValueError('hybrid requires C=64, safe_gate, gate, saved caches, exp2 and K-first state')
    if initial_state is not None or dht is not None:
        raise ValueError('hybrid does not support initial_state/dht')
    if allow_neg_eigval and (not use_beta_sigmoid_in_kernel):
        raise ValueError('allow_neg_eigval requires use_beta_sigmoid_in_kernel')
    if implementation not in ('auto', 'hybrid'):
        raise ValueError('implementation must be auto or hybrid')
    if (q_rstd is None) != (k_rstd is None):
        raise ValueError('q_rstd and k_rstd must be supplied together')
    if raw_g is None or A_log is None:
        raise ValueError('hybrid requires raw_g and A_log')
    heads, total = _runtime_shape(v, d_o)
    if any(t.shape[0] != heads for t in (q, k, Aqk, Akk, w, qg, kg, v_new)) or h.shape[1] != heads:
        raise ValueError('Q/K/V and all saved caches must have the same head count')
    if raw_g.shape != v.shape or beta_eff.shape != v.shape[:2]:
        raise ValueError('raw_g and beta_eff must use the saved NTD layout')
    if use_beta_sigmoid_in_kernel and beta_eff.dtype != torch.float32:
        raise ValueError('fused beta sigmoid requires FP32 beta_eff')
    return _run_hybrid(q, k, v, beta_eff, gk, Aqk, Akk, w, qg, kg, v_new,
        h, d_o, scale, heads, total, raw_g=raw_g, A_log=A_log, dt_bias=dt_bias,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound,
        q_rstd=q_rstd, k_rstd=k_rstd,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, prepared_metadata=prepared_metadata)

def _run_hybrid(q, k, v, beta_eff, gk, Aqk, Akk, w, qg, kg, v_new,
        h, d_o, scale, heads, total, *, raw_g, A_log, dt_bias=None,
        cu_seqlens=None, chunk_indices=None, lower_bound=-5.0,
        q_rstd=None, k_rstd=None, use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False, prepared_metadata=None):
    """Execute validated packed caches with runtime H/T and call-owned buffers."""
    with torch.npu.device(q.device):
        if total == 0:
            dq, dk, dv = (torch.zeros_like(v) for _ in range(3))
            dbias = None if dt_bias is None else torch.zeros_like(dt_bias)
            return (dq, dk, dv, torch.zeros_like(beta_eff), torch.zeros_like(raw_g), None, torch.zeros_like(A_log), dbias)
        cu, indices, cu_dev, idx_dev, off_dev, active = prepared_metadata if prepared_metadata is not None else _device_metadata(cu_seqlens, chunk_indices, q.device)
        if cu_dev.device != q.device:
            raise ValueError('prepared_metadata must use the input device')
        if cu[-1] != total:
            raise ValueError('cu_seqlens must end at the total token count')
        da = Aqk.new_empty(heads, total, 64, dtype=torch.float32)
        dv_local = v_new.new_empty(heads, total, 128)
        dqr = Aqk.new_empty(heads, total, 128, dtype=torch.float32)
        cores, offset = _k1_config(os.environ['ASCEND_HOME_PATH'], q.device.index)
        _prepare_launcher[cores,](v_new=v_new, A=Aqk, do=d_o, h=h, dv=dv_local, dA=da, dq_raw=dqr, cu_seqlens=cu_dev, chunk_indices=idx_dev, scale=scale, T_total=total, task_num=(len(indices) // 2) * (heads // 2), num_core=cores, HV=heads, V=128, K=128, BT=64, STATE_V_FIRST=False, CACHE_OFFSET=offset, DO_NTD=False, num_warps=1, num_stages=2)
        dh, _, scan = kda_bwd_state(qg, kg, w, gk, d_o, dv_local, cu_dev, off_dev, len(indices) // 2, scale, do_layout='TND', active_sequences=active, inplace_dv=True, head_window=_k2_task_shape(cu, cores, heads)[0])
        bias = torch.zeros(heads, 128, dtype=torch.float32, device=q.device) if dt_bias is None else dt_bias.reshape(heads, 128)
        from ._native_k3 import _finalize_ntd
        dq, dk, dv, db, dg, da_log, dbias = _finalize_ntd(q, k, v, gk,
            raw_g, beta_eff, A_log.reshape(heads), bias, Akk, v_new, h, dh,
            scan, da, dqr, scale=scale, lower_bound=lower_bound,
            q_rstd=q_rstd, k_rstd=k_rstd, cu_seqlens=cu, chunk_indices=indices,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval)
        if dt_bias is not None:
            dbias = dbias.reshape(dt_bias.shape)
        else:
            dbias = None
        return (dq, dk, dv, db, dg, None, da_log, dbias)
