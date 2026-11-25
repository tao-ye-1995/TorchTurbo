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

"""K2 buffer interfaces and compiler configuration."""
import triton.language as tl


@tl.core.builtin
def _add_buffer(lhs, rhs, dst, _semantic=None):
    """Compatible input buffer shapes; FP32 output uses dst storage."""
    _semantic.builder.create_dsa_add(lhs.handle, rhs.handle, dst.handle)


@tl.core.builtin
def _mul_buffer(lhs, rhs, dst, _semantic=None):
    """Compatible input buffer shapes; FP32 output uses dst storage."""
    _semantic.builder.create_dsa_mul(lhs.handle, rhs.handle, dst.handle)


@tl.core.builtin
def _sub_buffer(lhs, rhs, dst, _semantic=None):
    """Compatible input buffer shapes; FP32 output uses dst storage."""
    _semantic.builder.create_dsa_sub(lhs.handle, rhs.handle, dst.handle)


COMPILER_OPTIONS = dict(num_stages=1, multibuffer=False, enable_ubuf_saving=True,
    unit_flag=True, enable_auto_bind_sub_block=False, disable_auto_inject_block_sync=True,
    sync_solver=False)
