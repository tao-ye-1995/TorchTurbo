# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""AscendC KDA forward with FlagTree K1/K2 and bundled AscendC K3 backward."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import platform
from typing import Any, ClassVar

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImplKind
from ...registry import register_op
from .kda import AscendCKDA, ChunkKDAFunction


@lru_cache(maxsize=None)
def _configure_hybrid(cann: str) -> None:
    from ._kda_hybrid._environment import configure

    configure()


@lru_cache(maxsize=None)
def _is_supported_device(device: torch.device) -> bool:
    if device.type != "npu":
        return False
    return torch.npu.get_device_name(device).startswith("Ascend910_93")


def _backward_from_saved_tensors(
    ctx: Any,
    grad_output: torch.Tensor,
    kernel: Callable[..., tuple[torch.Tensor | None, ...]],
) -> tuple[torch.Tensor | None, ...]:
    """Apply a packed backward kernel and restore the model gradient views."""
    (
        q, k, v, raw_g, beta_eff, A_log, dt_bias, gk, aqk, akk, w,
        qg, kg, v_new, h, q_rstd, k_rstd,
    ) = ctx.saved_tensors
    # A contiguous TND gradient needs no layout copy. Expanded or strided
    # gradients still need contiguous storage for the kernel's strides.
    d_o = grad_output.squeeze(0).contiguous()
    dq, dk, dv, dbeta, dg, _, dA, dbias = kernel(
        q, k, v, beta_eff, gk, aqk, akk, w, qg, kg, v_new, h,
        d_o, ctx.scale,
        raw_g=raw_g, A_log=A_log, dt_bias=dt_bias,
        q_rstd=q_rstd, k_rstd=k_rstd,
        cu_seqlens=ctx.cu_seqlens, chunk_indices=ctx.chunk_indices,
        lower_bound=ctx.lower_bound,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
    )
    # K3 consumes and produces head-major NTD tensors; the model's gate and
    # beta inputs are token-major, just as in the original AscendC wrapper.
    dg = dg.transpose(0, 1).unsqueeze(0).to(ctx.g_dtype)
    dbeta = dbeta.transpose(0, 1).unsqueeze(0).to(ctx.beta_dtype)
    return (
        dq, dk, dv, dg, dbeta, dA, dbias,
        None, None, None, None, None, None, None, None,
    )


class HybridChunkKDAFunction(ChunkKDAFunction):
    """Preserve the AscendC forward caches and read incoming TND dO directly."""

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
        grad_final_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if grad_final_state is not None and bool(grad_final_state.any()):
            raise NotImplementedError(
                "KDA does not support gradients through final_state."
            )
        from ._kda_hybrid.hybrid import npu_chunk_kda_bwd_hybrid

        return _backward_from_saved_tensors(ctx, grad_output, npu_chunk_kda_bwd_hybrid)


@register_op(
    op_name="kda",
    impl_name="flagetree_kda_hybrid",
    kind=OpImplKind.CUSTOM,
    priority=200,
    devices=(TurboDevice.NPU,),
)
class FlagTreeHybridKDA(AscendCKDA):
    """Packed BF16 KDA with equal positive even head counts and K=V=128."""

    autograd_function: ClassVar[type[ChunkKDAFunction]] = HybridChunkKDAFunction

    def available(self) -> bool:
        # Probe optional dependencies without configuring process state.
        if platform.system() != "Linux" or platform.machine() != "aarch64":
            return False
        home = Path(os.environ.get("ASCEND_HOME_PATH", ""))
        info = home / "share/info/runtime/version.info"
        if not info.is_file() or "Version=9.1.0" not in info.read_text().splitlines():
            return False
        if not (home / "tools/bishengir/bin/bishengir-compile").is_file():
            return False
        native = Path(__file__).parent / "_kda_hybrid/_native"
        if not (native / "libkda_hybrid.so").is_file():
            return False
        if not (native / "opp/vendors/kda_hybrid/op_api/lib/libcust_opapi.so").is_file():
            return False
        try:
            import triton.experimental.tle  # noqa: F401
            import triton.language.extra.cann.extension  # noqa: F401
            import triton.spec.ascend.extension.buffer.language  # noqa: F401
        except ImportError:
            return False
        return super().available()

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
        if cu_seqlens is None or initial_state is not None:
            return False
        if not safe_gate or not use_gate_in_kernel or A_log is None:
            return False
        if q.ndim != 3 or g.ndim != 4 or beta.ndim != 3:
            return False
        batch, total, heads, width = g.shape
        if batch != 1 or total == 0 or heads <= 0 or heads % 2 or width != 128:
            return False
        if any(t.shape != (1, total, heads * 128) for t in (q, k, v)):
            return False
        if beta.shape != (1, total, heads) or A_log.numel() != heads:
            return False
        if dt_bias is not None and dt_bias.numel() != heads * 128:
            return False
        if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
            return False
        if g.dtype not in (torch.bfloat16, torch.float32):
            return False
        if beta.dtype not in (torch.bfloat16, torch.float32):
            return False
        tensors = (q, k, v, g, beta, A_log)
        if any(t.device != q.device for t in tensors):
            return False
        if dt_bias is not None and dt_bias.device != q.device:
            return False
        if cu_seqlens.device.type != "cpu" and cu_seqlens.device != q.device:
            return False
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            return False
        return _is_supported_device(q.device)

    def _prepare_execution(self) -> None:
        # Set up the bundled OPP before forward can initialize AscendC kernels.
        _configure_hybrid(os.environ["ASCEND_HOME_PATH"])
