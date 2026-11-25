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

"""K2 interface: q/k/w/dv BF16 [H,T,128], gk FP32 [H,T,128], dO BF16 TND/NTD. Outputs dh
BF16 [NC,H,128,128] and dv_scan BF16 [H,T,128].
"""
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.language.extra.cann.extension as al
import triton.spec.ascend.extension.buffer.language as bl
from ._state_common import _add_buffer, _mul_buffer, _sub_buffer

P = al.PIPE

@triton.jit(do_not_specialize=['T_total', 'task_num', 'num_core', 'HEADS'])
def _kda_bwd_state_kernel(
    q, k, w, gk, do, dv, dh, dv2, cu_seqlens, chunk_offsets,
    q_ws, w_ws, scale, T_total, task_num, num_core, HEADS,
    CACHE_OFFSET: tl.constexpr, DO_NTD: tl.constexpr,
    WINDOW: tl.constexpr, VBLOCK: tl.constexpr, INPLACE_DV: tl.constexpr,
):
    """cu_seqlens int32 [N+1]; chunk_offsets int64 [N+1]. q_ws/w_ws BF16
    [slots,128,VBLOCK]. WINDOW is 1/2/4 heads; VBLOCK is 32/64/128 columns. DO_NTD
    selects dO [H,T,V] instead of [T,H,V]. INPLACE_DV selects shared dv/dv2 output
    storage.
    """
    KEY: tl.constexpr = 128
    VALUE: tl.constexpr = 128
    CHUNK: tl.constexpr = 64
    ROWS: tl.constexpr = 32


    COOPERATIVE: tl.constexpr = WINDOW == 1
    STATE_ROWS: tl.constexpr = KEY // 2 if COOPERATIVE else KEY
    HEAD_GROUPS = HEADS // WINDOW
    V_GROUPS: tl.constexpr = VALUE // VBLOCK
    DO_PAIR: tl.constexpr = not DO_NTD and WINDOW >= 2 and VBLOCK == VALUE


    q = q + tl.full((), CACHE_OFFSET // (q.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    k = k + tl.full((), CACHE_OFFSET // (k.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    w = w + tl.full((), CACHE_OFFSET // (w.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    do = do + tl.full((), CACHE_OFFSET // (do.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    dv = dv + tl.full((), CACHE_OFFSET // (dv.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    gk = gk + tl.full((), CACHE_OFFSET // (gk.dtype.element_ty.primitive_bitwidth // 8), tl.int64)
    core = tl.program_id(0)
    ready = 0
    result_ready = 1

    with tle.scope(core_mode='cube'):
        if DO_PAIR:
            do_l1 = tle.dsa.to_tensor(tle.dsa.alloc(
                [CHUNK, 2 * VALUE], dtype=do.dtype.element_ty,
                mem_addr_space=tle.dsa.ascend.L1), writable=True)
        for task in tl.range(core, task_num, num_core):

            seq = task // (HEAD_GROUPS * V_GROUPS)
            group = task % (HEAD_GROUPS * V_GROUPS)
            head_base = (group // V_GROUPS) * WINDOW
            col = (group % V_GROUPS) * VBLOCK
            bos = tl.load(cu_seqlens + seq).to(tl.int64)
            eos = tl.load(cu_seqlens + seq + 1).to(tl.int64)
            length = (eos - bos).to(tl.int32)
            boh = tl.load(chunk_offsets + seq).to(tl.int64)


            for it in range(tl.cdiv(length, CHUNK) - 1, -1, -1):
                row = it * CHUNK
                for owner in tl.static_range(WINDOW):
                    hh = tl.cast(head_base + owner, tl.int64)
                    base = hh * T_total + bos
                    slot = core * WINDOW + owner

                    kt = tl.load(tl.make_block_ptr(k + base * KEY,
                        (length, KEY), (KEY, 1), (row, 0), (CHUNK, KEY), (1, 0)),
                        boundary_check=(0,), padding_option='zero')
                    al.sync_block_wait('vector', 'cube', ready, P.PIPE_MTE3, P.PIPE_MTE2)
                    if it + 1 < tl.cdiv(length, CHUNK):
                        state_tile = tl.load(tl.make_block_ptr(
                            dh + ((boh + it) * HEADS + hh) * KEY * VALUE,
                            (KEY, VALUE), (VALUE, 1), (0, col), (KEY, VBLOCK), (1, 0)))
                        kd = tl.dot(kt, state_tile, allow_tf32=False)

                        token = row + tl.arange(0, CHUNK)
                        columns = col + tl.arange(0, VBLOCK)
                        tl.atomic_add(dv2 + (base + token[:, None]) * VALUE + columns[None, :],
                                      kd.to(dv2.dtype.element_ty), mask=token[:, None] < length, sem='relaxed')

                    if DO_NTD:
                        do_tile = tl.load(tl.make_block_ptr(do + base * VALUE,
                            (length, VALUE), (VALUE, 1), (row, col), (CHUNK, VBLOCK), (1, 0)),
                            boundary_check=(0,), padding_option='zero')
                    elif DO_PAIR:

                        if owner % 2 == 0:
                            do_values = tl.load(tl.make_block_ptr(do + (bos * HEADS + hh) * VALUE,
                                (length, 2 * VALUE), (HEADS * VALUE, 1), (row, 0),
                                (CHUNK, 2 * VALUE), (1, 0)), boundary_check=(0,), padding_option='zero')
                            do_group = tle.dsa.insert_slice(do_l1, do_values,
                                offsets=[0, 0], sizes=[CHUNK, 2 * VALUE], strides=[1, 1])
                        do_tile = tle.dsa.extract_slice(do_group,
                            offsets=[0, (owner % 2) * VALUE], sizes=[CHUNK, VALUE], strides=[1, 1])
                    else:

                        do_tile = tl.load(tl.make_block_ptr(do + (bos * HEADS + hh) * VALUE,
                            (length, VALUE), (HEADS * VALUE, 1), (row, col),
                            (CHUNK, VBLOCK), (1, 0)), boundary_check=(0,), padding_option='zero')
                    qt = tl.load(tl.make_block_ptr(q + base * KEY,
                        (KEY, length), (1, KEY), (0, row), (KEY, CHUNK), (0, 1)),
                        boundary_check=(1,), padding_option='zero')
                    qa = tl.dot(qt, do_tile, allow_tf32=False)
                    tl.store(tl.make_block_ptr(q_ws + slot * KEY * VBLOCK,
                        (KEY, VBLOCK), (VBLOCK, 1), (0, 0), (KEY, VBLOCK), (1, 0)),
                        qa.to(q_ws.dtype.element_ty))

                    wt = tl.load(tl.make_block_ptr(w + base * KEY,
                        (KEY, length), (1, KEY), (0, row), (KEY, CHUNK), (0, 1)),
                        boundary_check=(1,), padding_option='zero')
                    dt = tl.load(tl.make_block_ptr(dv2 + base * VALUE,
                        (length, VALUE), (VALUE, 1), (row, col), (CHUNK, VBLOCK), (1, 0)),
                        boundary_check=(0,), padding_option='zero')
                    wa = tl.dot(wt, dt, allow_tf32=False)
                    tl.store(tl.make_block_ptr(w_ws + slot * KEY * VBLOCK,
                        (KEY, VBLOCK), (VBLOCK, 1), (0, 0), (KEY, VBLOCK), (1, 0)),
                        wa.to(w_ws.dtype.element_ty))
                    al.sync_block_set('cube', 'vector', result_ready, P.PIPE_FIX, P.PIPE_MTE2)

    with tle.scope(core_mode='vector'):
        sub = al.sub_vec_id()

        state_a = bl.to_buffer(tl.full((STATE_ROWS, VBLOCK), 0, tl.float32), space=al.ascend_address_space.UB)
        if WINDOW == 4:
            state_b = bl.to_buffer(tl.full((STATE_ROWS, VBLOCK), 1, tl.float32), space=al.ascend_address_space.UB)
        if COOPERATIVE:
            key_begin = sub.to(tl.int32) * (KEY // 2)
            key_end = key_begin + KEY // 2
            token_begin = sub.to(tl.int32) * (CHUNK // 2)
            token_end = token_begin + CHUNK // 2
        else:
            key_begin = 0
            key_end = KEY
            token_begin = 0
            token_end = CHUNK
        for task in tl.range(core, task_num, num_core):


            seq = task // (HEAD_GROUPS * V_GROUPS)
            group = task % (HEAD_GROUPS * V_GROUPS)
            head_base = (group // V_GROUPS) * WINDOW
            col = (group % V_GROUPS) * VBLOCK
            bos = tl.load(cu_seqlens + seq).to(tl.int64)
            eos = tl.load(cu_seqlens + seq + 1).to(tl.int64)
            length = (eos - bos).to(tl.int32)
            boh = tl.load(chunk_offsets + seq).to(tl.int64)
            for it in range(tl.cdiv(length, CHUNK) - 1, -1, -1):
                row = it * CHUNK
                last = tl.minimum((it + 1) * CHUNK, length) - 1

                for owner in tl.static_range(WINDOW):
                    if COOPERATIVE or sub == owner % 2:
                        hh = tl.cast(head_base + owner, tl.int64)
                        base = hh * T_total + bos
                        if not INPLACE_DV:

                            for dr in tl.range(token_begin, token_end, ROWS):
                                raw = tl.load(tl.make_block_ptr(dv + base * VALUE,
                                    (length, VALUE), (VALUE, 1), (row + dr, col), (ROWS, VBLOCK), (1, 0)),
                                    boundary_check=(0,), padding_option='zero')
                                tl.store(tl.make_block_ptr(dv2 + base * VALUE,
                                    (length, VALUE), (VALUE, 1), (row + dr, col), (ROWS, VBLOCK), (1, 0)),
                                    raw, boundary_check=(0,))
                        for kr in tl.range(key_begin, key_end, ROWS):
                            hp = tl.make_block_ptr(dh + ((boh + it) * HEADS + hh) * KEY * VALUE,
                                (KEY, VALUE), (VALUE, 1), (kr, col), (ROWS, VBLOCK), (1, 0))
                            if it + 1 == tl.cdiv(length, CHUNK):
                                tl.store(hp, tl.full((ROWS, VBLOCK), 0, dh.dtype.element_ty))
                            else:
                                if owner < 2:
                                    state_row = bl.subview(state_a, offsets=[kr - key_begin, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                else:
                                    state_row = bl.subview(state_b, offsets=[kr - key_begin, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                state = bl.to_tensor(state_row, writable=True)
                                tl.store(hp, state.to(dh.dtype.element_ty))

                    al.sync_block_set('vector', 'cube', ready, P.PIPE_MTE3, P.PIPE_MTE2)
                    if COOPERATIVE or sub == owner % 2:
                        if it + 1 < tl.cdiv(length, CHUNK):
                            base = tl.cast(head_base + owner, tl.int64) * T_total + bos
                            factors = tl.exp2(tl.load(gk + (base + last) * KEY + tl.arange(0, KEY)))
                            for kr in tl.range(key_begin, key_end, ROWS):
                                if owner < 2:
                                    state_row = bl.subview(state_a, offsets=[kr - key_begin, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                else:
                                    state_row = bl.subview(state_b, offsets=[kr - key_begin, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                factor = tl.gather(factors, kr + tl.arange(0, ROWS), 0)
                                factor_full = tl.broadcast_to(factor[:, None], (ROWS, VBLOCK))
                                factor_buf = bl.to_buffer(factor_full, space=al.ascend_address_space.UB)
                                factor_view = bl.subview(factor_buf, offsets=[0, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                _mul_buffer(state_row, factor_view, state_row)

                for owner in tl.static_range(WINDOW):
                    al.sync_block_wait('cube', 'vector', result_ready, P.PIPE_FIX, P.PIPE_MTE2)
                    if COOPERATIVE or sub == owner % 2:
                        slot = core * WINDOW + owner
                        for kr in tl.range(key_begin, key_end, ROWS):

                            qa_raw = tl.load(tl.make_block_ptr(q_ws + slot * KEY * VBLOCK,
                                (KEY, VBLOCK), (VBLOCK, 1), (kr, 0), (ROWS, VBLOCK), (1, 0)))
                            wa_raw = tl.load(tl.make_block_ptr(w_ws + slot * KEY * VBLOCK,
                                (KEY, VBLOCK), (VBLOCK, 1), (kr, 0), (ROWS, VBLOCK), (1, 0)))
                            qa = qa_raw.to(tl.float32)
                            wa = wa_raw.to(tl.float32)
                            if owner < 2:
                                state_row = bl.subview(state_a, offsets=[kr - key_begin, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                            else:
                                state_row = bl.subview(state_b, offsets=[kr - key_begin, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                            if it + 1 == tl.cdiv(length, CHUNK):
                                qscaled = bl.to_buffer(qa * scale, space=al.ascend_address_space.UB)
                                wfloat = bl.to_buffer(wa, space=al.ascend_address_space.UB)
                                qsview = bl.subview(qscaled, offsets=[0, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                wfview = bl.subview(wfloat, offsets=[0, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                _sub_buffer(qsview, wfview, state_row)
                            else:
                                increment = qa * scale - wa
                                inc_buf = bl.to_buffer(increment, space=al.ascend_address_space.UB)
                                inc_view = bl.subview(inc_buf, offsets=[0, 0], sizes=[ROWS, VBLOCK], strides=[1, 1])
                                _add_buffer(state_row, inc_view, state_row)

