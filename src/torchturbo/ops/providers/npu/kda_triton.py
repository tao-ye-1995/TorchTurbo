# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""AscendC KDA forward with FlagTree K1/K2 and precompiled Triton K3 backward."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
from typing import Any, ClassVar

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImplKind
from ...registry import register_op
from .kda import AscendCKDA, ChunkKDAFunction
from .kda_hybrid import FlagTreeHybridKDA, _backward_from_saved_tensors


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    path = Path(__file__).parent / "_kda_triton/_ir/manifest.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def _compiler_version(compiler: str) -> str:
    return subprocess.check_output([compiler, "--version"], text=True)


@lru_cache(maxsize=None)
def _configure_triton(cann: str) -> None:
    from ._kda_triton._environment import configure

    configure()


@lru_cache(maxsize=None)
def _is_supported_device(device: torch.device) -> bool:
    return device.type == "npu" and torch.npu.get_device_name(device) == _manifest()["target"]


class TritonChunkKDAFunction(ChunkKDAFunction):
    """Use the original forward with a call-owned all-Triton backward pipeline."""

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_final_state is not None and bool(grad_final_state.any()):
            raise NotImplementedError("KDA does not support gradients through final_state.")
        from ._kda_triton.backward import npu_chunk_kda_bwd_triton

        return _backward_from_saved_tensors(ctx, grad_output, npu_chunk_kda_bwd_triton)


@register_op(
    op_name="kda",
    impl_name="flagetree_kda_triton",
    kind=OpImplKind.CUSTOM,
    priority=300,
    devices=(TurboDevice.NPU,),
)
class FlagTreeTritonKDA(FlagTreeHybridKDA):
    """Packed KDA with the frozen FP32 gate, sigmoid, and normalization ABI."""

    autograd_function: ClassVar[type[ChunkKDAFunction]] = TritonChunkKDAFunction

    def available(self) -> bool:
        if platform.system() != "Linux" or platform.machine() != "aarch64":
            return False
        assets = Path(__file__).parent / "_kda_triton/_ir"
        if not (assets / "manifest.json").is_file():
            return False
        manifest = _manifest()
        required = ("_k3_torchturbo_native_tail.npubin", "_k3_torchturbo_native_tail.json")
        if any(not (assets / "dynamic" / name).is_file() for name in required):
            return False
        home = Path(os.environ.get("ASCEND_HOME_PATH", ""))
        info = home / "share/info/runtime/version.info"
        version = f"Version={manifest['cann_version']}"
        if not info.is_file() or version not in info.read_text().splitlines():
            return False
        compiler = home / "tools/bishengir/bin/bishengir-compile"
        if not compiler.is_file() or _compiler_version(str(compiler)) != manifest["compiler_version"]:
            return False
        try:
            import triton
            import triton.experimental.tle  # noqa: F401
            import triton.language.extra.cann.extension  # noqa: F401
            import triton.spec.ascend.extension.buffer.language  # noqa: F401
            from triton.compiler.compiler import ASTSource, CompiledKernel  # noqa: F401
        except ImportError:
            return False
        # The forward still requires fla_npu. Backward needs no hybrid K3 library.
        return triton.__version__ == manifest["triton_version"] and AscendCKDA.available(self)

    def supports(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = True,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> bool:
        if not super().supports(
            q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel, safe_gate=safe_gate,
            lower_bound=lower_bound, cu_seqlens=cu_seqlens,
        ):
            return False
        if g.dtype != torch.float32 or not use_qk_l2norm_in_kernel or not use_beta_sigmoid_in_kernel:
            return False
        if scale is not None and scale != _manifest()["scale"]:
            return False
        if lower_bound is not None and lower_bound != _manifest()["lower_bound"]:
            return False
        # Float casts in the common forward make mixed-precision parameters
        # contiguous. Existing FP32 parameters must already have linear storage.
        for parameter in (A_log, dt_bias):
            if parameter is not None and parameter.dtype == torch.float32 and not parameter.is_contiguous():
                return False
        total, heads = g.shape[1:3]
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences <= 0 or heads * total * 128 > 2**31 - 1:
            return False
        # Bound NC from shapes without copying device boundaries to the host.
        # For valid packed lengths, sum(ceil(length / 64)) <= this bound.
        max_chunks = min(total, (total + 63 * num_sequences) // 64)
        if heads * max_chunks * 16384 > 2**31 - 1:
            return False
        return _is_supported_device(q.device)

    def _prepare_execution(self) -> None:
        _configure_triton(os.environ["ASCEND_HOME_PATH"])
