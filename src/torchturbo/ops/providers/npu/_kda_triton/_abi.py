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

"""K3 runtime T/H/NC launch signature."""
import triton
import triton.language as tl
BF16={24,28,29,30}


@triton.jit
def _dynamic_abi(Q,K,V,G,VN,HSTATE,DH,DS,AKK,BETA,RAW,ALOG,BIAS,QR,KR,DA,DQR,CU,IDX,STATE,DVB,DW,ZV,GB,ZW,KE,ZB,TZA,DAK,LOW,LK,KN,QP,KP,LOCAL,LEFT,RIGHT,CONST,DWFP,TZAFP,EG,DQ,DK,DV,DB,DG,PA,PB,DALOG,DBIAS,T,H,NC,CORES:tl.constexpr,SCALE:tl.constexpr,COMPACT,FAST_ROUND:tl.constexpr,SHARE_EXP:tl.constexpr,PRUNE_BANDS:tl.constexpr,CACHE_EXP:tl.constexpr,OVERLAP:tl.constexpr,CACHE_OFFSET,SIGMOID:tl.constexpr,LOWER_BOUND:tl.constexpr,STABLE_GATE:tl.constexpr,L0C_TAG:tl.constexpr):
    pass
