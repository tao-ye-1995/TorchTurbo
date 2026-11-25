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

# Upstream portions from flash-linear-attention-npu:
# Copyright (c) 2026 Tianjin University
# Licensed under the BSD 3-Clause License; see LICENSES/LICENSE_FLA_NPU.

"""AscendC K3 tensor interface; packed inputs and gradients default to NTD."""
import ctypes
from functools import lru_cache
from pathlib import Path
import threading
import torch
import torch_npu

_op_handle = None
_library = None
_stream_accessor = None
_load_lock = threading.Lock()


def _op():
    global _op_handle, _library, _stream_accessor
    if _op_handle is None:
        with _load_lock:
            if _op_handle is None:
                path = Path(__file__).parent / '_native/libkda_hybrid.so'
                torch.ops.load_library(str(path))
                _library = ctypes.CDLL(str(path))
                probe = _library.kda_hybrid_queue_available
                probe.restype = ctypes.c_int32
                if probe() and hasattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait'):
                    _stream_accessor = torch_npu._C._npu_getCurrentRawStreamNoWait
                else:
                    _stream_accessor = torch_npu._C._npu_getCurrentRawStream
                _op_handle = torch.ops.kda_hybrid_native.npu_chunk_kda_bwd_finalize
    return _op_handle


def _current_stream_ptr():

    return int(_stream_accessor(torch.npu.current_device()))


@lru_cache(maxsize=64)
def _host_tuple(values):
    return torch.tensor(values, dtype=torch.int64, device='cpu')






def _finalize_ntd(q, k, v, gk, raw_g, beta, a_log, dt_bias, akk, v_new,
        h, dh, dv_scan, d_aqk, dq_raw, *, scale, lower_bound, q_rstd, k_rstd,
        cu_seqlens, chunk_indices, use_beta_sigmoid_in_kernel, allow_neg_eigval):
    """Validated packed NTD caches; standalone output_layout=NTD, model_layout=''."""
    op = _op()
    return op(q, k, v, gk, raw_g, beta, a_log, dt_bias, akk, v_new, h, dh,
        dv_scan, d_aqk, dq_raw, q_rstd, k_rstd, _host_tuple(cu_seqlens),
        _host_tuple(chunk_indices), float(scale), float(lower_bound), 64,
        True, True, True, False, bool(use_beta_sigmoid_in_kernel),
        bool(allow_neg_eigval), 3, _current_stream_ptr(), 4)
