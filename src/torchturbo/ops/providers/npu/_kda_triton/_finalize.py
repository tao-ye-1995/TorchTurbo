"""Launch runtime-length high-precision K3 IR with call-owned device storage."""
from functools import lru_cache
import json
from pathlib import Path

import torch
from triton.compiler.compiler import ASTSource, CompiledKernel

from ._abi import _dynamic_abi, BF16

ASSETS = Path(__file__).resolve().parent / '_ir'
MANIFEST = json.loads((ASSETS / 'manifest.json').read_text())
WORKSPACE_BYTES = MANIFEST['workspace_bytes']
# Byte offsets within each 368 KiB core/owner/bank slot.
OFFSETS = {22: 0, 23: 32768, 25: 311296, 26: 65536, 27: 327680, 28: 344064, 29: 360448, 31: 98304, 32: 344064, 33: 360448, 34: 114688, 35: 147456, 36: 180224, 37: 212992, 38: 245760, 39: 278528, 43: 311296}
WORKSPACE_VIEWS = tuple((OFFSETS.get(arg, 0), (WORKSPACE_BYTES - OFFSETS.get(arg, 0)) // (2 if arg in BF16 else 4)) for arg in range(22, 44))


class _WorkspacePointer(int):
    """Device address with element count for the launcher's profiling metadata."""
    def __new__(cls, address, elements):
        pointer = int.__new__(cls, address)
        pointer.elements = elements
        return pointer

    def size(self):
        return (self.elements,)




@lru_cache(maxsize=8)
def _dynamic_kernel(device):
    folder=ASSETS/'dynamic'
    binary=folder/'_k3_torchturbo_native_tail.npubin'
    types=['*bf16']*19 + ['*bf16' if i in BF16 else '*fp32' for i in range(22,44)] + ['*bf16']*3 + ['*fp32']*6
    for i in [3,9,10,11,12,13,14,15,16]:types[i]='*fp32'
    for i in [17,18]:types[i]='*i32'
    constants={53:24,54:MANIFEST['scale'],**{i:True for i in range(55,61)},
               61:0,62:True,63:-5.,64:True,65:'dynamic_ir'}
    signature={n:types[i] if i<50 else ('i32' if i in (50,51,52) else 'constexpr')
               for i,n in enumerate(_dynamic_abi.arg_names)}
    source=ASTSource(_dynamic_abi,signature,{(i,):v for i,v in constants.items()})
    metadata=folder/'_k3_torchturbo_native_tail.json'
    meta=json.loads(metadata.read_text())
    kernel = CompiledKernel(source,{'_k3_torchturbo_native_tail.npubin':str(binary),
                                 '_k3_torchturbo_native_tail.json':str(metadata)},meta['hash'])
    return kernel[(24,1,1)]

def kda_bwd_finalize(q,k,v,gk,raw,beta,alog,bias,akk,vn,h,dh,scan,da,dqr,cu_dev,idx_dev,
                     *,scale,q_rstd,k_rstd,n_chunks):
    """Return seven NTD gradients; main workspace is 34.5 MiB per call."""
    heads,total,_=q.shape
    arena=torch.empty(WORKSPACE_BYTES//4,device=q.device,dtype=torch.float32)
    base=arena.data_ptr()
    ws=[_WorkspacePointer(base+offset, elements) for offset, elements in WORKSPACE_VIEWS]
    outputs=tuple(torch.empty_like(t) for t in [q,k,v,beta,raw,alog,bias])
    pa=torch.empty((heads,n_chunks),device=q.device,dtype=torch.float32)
    pb=torch.empty((n_chunks,heads,128),device=q.device,dtype=torch.float32)
    args=[q,k,v,gk,vn,h,dh,scan,akk,beta,raw,alog,bias,q_rstd,k_rstd,da,dqr,cu_dev,idx_dev]
    args.extend(ws);args.extend([*outputs[:5],pa,pb,*outputs[5:]])
    args.extend([total,heads,n_chunks,24,scale,True,True,True,True,True,True,0,True,-5.,True,'dynamic_ir'])
    _dynamic_kernel(q.device.index)(*args)
    return outputs
