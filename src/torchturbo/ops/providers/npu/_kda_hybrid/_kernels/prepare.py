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

"""K1 interface: A [H,T,64] BF16; v_new [H,T,128] BF16; dO TND/NTD BF16; state [NC,H,K,V] BF16.
Outputs dv_new [H,T,128] BF16, dA [H,T,64] FP32 and dq [H,T,128] FP32.
"""
import triton
import triton.language as tl
import triton.experimental.tle as tle

from ._prepare_launch import _PrepareLauncher


@triton.jit
def _compute_da_dq(d, v, state, da, dq, head, bos, row0, seq_len, T_total, scale, V: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    base = head.to(tl.int64) * T_total + bos
    out = tl.dot(d, tl.trans(v), out_dtype=tl.float32)
    r = tl.arange(0, BT)
    valid = row0 + r < seq_len
    out = tl.where((r[:, None] >= r[None, :]) & valid[:, None] & valid[None, :], out * scale, 0.0)
    tl.store(tl.make_block_ptr(da + base * BT, (seq_len, BT), (BT, 1), (row0, 0), (BT, BT), (1, 0)), out.to(da.dtype.element_ty), boundary_check=(0,))
    out = tl.dot(d, state.to(d.dtype), out_dtype=tl.float32)
    tl.store(tl.make_block_ptr(dq + base * K, (seq_len, K), (K, 1), (row0, 0), (BT, K), (1, 0)), out.to(dq.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _compute_head(a, d, v, state, dv, da, dq, head, bos, row0, seq_len, T_total, scale, V: tl.constexpr, K: tl.constexpr, BT: tl.constexpr):
    base = head.to(tl.int64) * T_total + bos
    out = tl.dot(tl.trans(a), d, out_dtype=tl.float32)
    tl.store(tl.make_block_ptr(dv + base * V, (seq_len, V), (V, 1), (row0, 0), (BT, V), (1, 0)), out.to(dv.dtype.element_ty), boundary_check=(0,))
    _compute_da_dq(d, v, state, da, dq, head, bos, row0, seq_len, T_total, scale, V, K, BT)


@triton.jit
def _load_a(A, head, bos, seq_len, row0, T_total, BT: tl.constexpr):
    base = head.to(tl.int64) * T_total + bos
    return tl.load(tl.make_block_ptr(A + base * BT, (seq_len, BT), (BT, 1),
                   (row0, 0), (BT, BT), (1, 0)), boundary_check=(0,), padding_option="zero")


@triton.jit
def _load_v_state(v_new, h, head, ic, bos, seq_len, row0, T_total,
                  HV, V: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
                  STATE_V_FIRST: tl.constexpr):
    base = head.to(tl.int64) * T_total + bos
    v = tl.load(tl.make_block_ptr(v_new + base * V, (seq_len, V), (V, 1),
                (row0, 0), (BT, V), (1, 0)), boundary_check=(0,), padding_option="zero")
    state_base = (ic.to(tl.int64) * HV + head) * K * V
    if STATE_V_FIRST:
        p_h = tl.make_block_ptr(h + state_base, (V, K), (K, 1), (0, 0), (V, K), (1, 0))
    else:
        p_h = tl.make_block_ptr(h + state_base, (V, K), (1, V), (0, 0), (V, K), (0, 1))
    return v, tl.load(p_h)


@triton.jit(do_not_specialize=['T_total', 'task_num', 'num_core', 'HV'])
def _kda_bwd_prepare_kernel(
    v_new, A, do, h, dv, dA, dq_raw, cu_seqlens, chunk_indices,
    scale, T_total, task_num, num_core,
    HV, V: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr, CACHE_OFFSET: tl.constexpr, DO_NTD: tl.constexpr = False,
):
    """cu_seqlens int32 [N+1]; chunk_indices int32 sequence/chunk pairs [NC,2]. HV: even
    head count. V=K=128, BT=64. STATE_V_FIRST selects [NC,H,V,K] instead of [NC,H,K,V].
    DO_NTD selects [H,T,V] instead of [T,H,V]. CACHE_OFFSET is a byte offset; runtime
    sizes and core counts are integer scalars.
    """
    if CACHE_OFFSET:
        v_new = v_new + tl.full((), CACHE_OFFSET // (v_new.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
        A = A + tl.full((), CACHE_OFFSET // (A.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
        do = do + tl.full((), CACHE_OFFSET // (do.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
        h = h + tl.full((), CACHE_OFFSET // (h.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    if DO_NTD:
        d0_l1 = tle.dsa.alloc([BT, V], dtype=do.dtype.element_ty, mem_addr_space=tle.dsa.ascend.L1)
        d1_l1 = tle.dsa.alloc([BT, V], dtype=do.dtype.element_ty, mem_addr_space=tle.dsa.ascend.L1)
    else:
        d_l1 = tle.dsa.alloc([BT, 2 * V], dtype=do.dtype.element_ty, mem_addr_space=tle.dsa.ascend.L1)
    groups = HV // 2
    for task in tl.range(tl.program_id(0), task_num, num_core):
        ic = task // groups
        head = (task % groups) * 2
        seq = tl.load(chunk_indices + ic * 2).to(tl.int32)
        chunk = tl.load(chunk_indices + ic * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + seq).to(tl.int64)
        eos = tl.load(cu_seqlens + seq + 1).to(tl.int64)
        seq_len = (eos - bos).to(tl.int32)
        row0 = chunk * BT
        a0 = _load_a(A, head, bos, seq_len, row0, T_total, BT)
        if DO_NTD:
            do_base = head.to(tl.int64) * T_total + bos
            d0 = tl.load(tl.make_block_ptr(do + do_base * V, (seq_len, V), (V, 1),
                         (row0, 0), (BT, V), (1, 0)), boundary_check=(0,), padding_option="zero")
            d1 = tl.load(tl.make_block_ptr(do + (do_base + T_total) * V, (seq_len, V), (V, 1),
                         (row0, 0), (BT, V), (1, 0)), boundary_check=(0,), padding_option="zero")
            d0 = tle.dsa.insert_slice(tle.dsa.to_tensor(d0_l1, writable=True), d0,
                                     offsets=[0, 0], sizes=[BT, V], strides=[1, 1])
            d1 = tle.dsa.insert_slice(tle.dsa.to_tensor(d1_l1, writable=True), d1,
                                     offsets=[0, 0], sizes=[BT, V], strides=[1, 1])
        else:
            do_base = bos * HV + head.to(tl.int64)
            do_heads = tl.load(tl.make_block_ptr(do + do_base * V, (seq_len, 2 * V), (HV * V, 1),
                           (row0, 0), (BT, 2 * V), (1, 0)), boundary_check=(0,), padding_option="zero")
            do_heads = tle.dsa.insert_slice(tle.dsa.to_tensor(d_l1, writable=True), do_heads,
                                       offsets=[0, 0], sizes=[BT, 2 * V], strides=[1, 1])
            d0 = tle.dsa.extract_slice(do_heads, offsets=[0, 0], sizes=[BT, V], strides=[1, 1])
            d1 = tle.dsa.extract_slice(do_heads, offsets=[0, V], sizes=[BT, V], strides=[1, 1])
        if DO_NTD:

            v0, state0 = _load_v_state(v_new, h, head, ic, bos, seq_len, row0,
                                      T_total, HV, V, K, BT, STATE_V_FIRST)
            _compute_head(a0, d0, v0, state0, dv, dA, dq_raw, head, bos, row0,
                          seq_len, T_total, scale, V, K, BT)
            a1 = _load_a(A, head + 1, bos, seq_len, row0, T_total, BT)
            v1, state1 = _load_v_state(v_new, h, head + 1, ic, bos, seq_len, row0,
                                      T_total, HV, V, K, BT, STATE_V_FIRST)
            _compute_head(a1, d1, v1, state1, dv, dA, dq_raw, head + 1, bos, row0,
                          seq_len, T_total, scale, V, K, BT)
        else:
            base0 = head.to(tl.int64) * T_total + bos
            dv0 = tl.dot(tl.trans(a0), d0, out_dtype=tl.float32)
            tl.store(tl.make_block_ptr(dv + base0 * V, (seq_len, V), (V, 1),
                     (row0, 0), (BT, V), (1, 0)), dv0.to(dv.dtype.element_ty), boundary_check=(0,))
            a1 = _load_a(A, head + 1, bos, seq_len, row0, T_total, BT)
            dv1 = tl.dot(tl.trans(a1), d1, out_dtype=tl.float32)
            tl.store(tl.make_block_ptr(dv + (base0 + T_total) * V, (seq_len, V), (V, 1),
                     (row0, 0), (BT, V), (1, 0)), dv1.to(dv.dtype.element_ty), boundary_check=(0,))

            for head_in_pair in tl.range(0, 2, loop_unroll_factor=1):
                current_head = head + head_in_pair
                d = tle.dsa.extract_slice(do_heads, offsets=[0, head_in_pair * V],
                                          sizes=[BT, V], strides=[1, 1])
                v, state = _load_v_state(v_new, h, current_head, ic, bos, seq_len, row0,
                                          T_total, HV, V, K, BT, STATE_V_FIRST)
                _compute_da_dq(d, v, state, dA, dq_raw, current_head, bos, row0,
                                seq_len, T_total, scale, V, K, BT)



_prepare_launcher = _PrepareLauncher(_kda_bwd_prepare_kernel)

