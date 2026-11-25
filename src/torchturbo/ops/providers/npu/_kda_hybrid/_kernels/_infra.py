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

"""Host device properties and L2 byte-offset queries."""
import ctypes
from functools import lru_cache
import os
from pathlib import Path

import torch
import torch_npu
import triton

def get_npu_properties():
    return triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())


def get_l2_bypass_offset(device_id):
    return _cached_l2_bypass_offset(os.environ['ASCEND_HOME_PATH'], int(device_id))


@lru_cache(None)
def _cached_l2_bypass_offset(cann, device_id):
    runtime = ctypes.CDLL(str(Path(cann).resolve() / 'lib64/libruntime.so'))
    query = runtime.rtGetL2CacheOffset
    query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int32
    offset = ctypes.c_uint64()
    rc = query(device_id, ctypes.byref(offset))
    if rc or not offset.value:
        raise RuntimeError(f'No L2 bypass offset for device {device_id}: rc={rc}')
    return offset.value

