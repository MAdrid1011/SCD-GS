"""
MaskedGaussianProxy — a lightweight shim that presents a subset of a
Gaussian model's attributes using torch.index_select (no data copy).

Used by GLASSPipeline.make_render_proxy() during FPS measurement to pass a
smaller Gaussian set to the baseline render() function without modifying
training or model state.

Compatibility
-------------
The proxy exposes the same property API used by all four baseline render()
functions and the diff-gaussian-rasterization CUDA kernel:

    .get_xyz          → [M, 3]   float32, activated positions
    .get_opacity      → [M, 1]   float32, activated opacity (sigmoid)
    .get_scaling      → [M, 3]   float32, activated scale (exp)
    .get_rotation     → [M, 4]   float32, normalised quaternion
    .get_features     → [M, 3, K] float32, SH features
    .max_radii2D      → [M]      float32, per-Gaussian max 2D radii
    .active_sh_degree → int      (forwarded from wrapped model)
    .max_sh_degree    → int      (forwarded from wrapped model)

All indexed tensors are detached (no gradient tracking) because the proxy
is only used during inference (FPS measurement), never during training.

Notes
-----
- index_select returns a new tensor backed by the same storage (copy-free for
  contiguous inputs), so the proxy has negligible memory overhead.
- The proxy does NOT support in-place writes or optimizer-related accessors;
  it is read-only.
"""
from __future__ import annotations

import torch


class MaskedGaussianProxy:
    """Read-only view into a Gaussian model, exposing only indexed entries.

    Works transparently with any Gaussian model that exposes the standard
    property API (3DGStream, QUEEN, HiCoM MergedGaussianModel, Swift4D).

    Parameters
    ----------
    gaussians : any Gaussian model object
        Source model. Must expose .get_xyz, .get_opacity, .get_scaling,
        .get_rotation, .get_features, .max_radii2D, .active_sh_degree,
        .max_sh_degree.
    indices : torch.LongTensor, shape [M]
        Indices of Gaussians to include. Must be on the same device as
        the model's tensors (typically CUDA).
    """

    def __init__(self, gaussians, indices: torch.Tensor) -> None:
        self._g   = gaussians
        self._idx = indices.long()
        if self._idx.device != torch.device("cpu"):
            # Keep indices on the same device as the model tensors
            pass
        self._n = int(self._idx.numel())

    # ── Internal selector ─────────────────────────────────────────────────────

    def _sel(self, t: torch.Tensor) -> torch.Tensor:
        """Select rows from a 2-D or higher tensor using self._idx."""
        idx = self._idx.to(t.device)
        return torch.index_select(t.detach(), 0, idx)

    # ── Standard Gaussian property API ───────────────────────────────────────

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._sel(self._g.get_xyz)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self._sel(self._g.get_opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self._sel(self._g.get_scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self._sel(self._g.get_rotation)

    @property
    def get_features(self) -> torch.Tensor:
        return self._sel(self._g.get_features)

    @property
    def max_radii2D(self) -> torch.Tensor:
        return self._sel(self._g.max_radii2D)

    # ── Scalar / non-indexed properties ──────────────────────────────────────

    @property
    def active_sh_degree(self) -> int:
        return self._g.active_sh_degree

    @property
    def max_sh_degree(self) -> int:
        return self._g.max_sh_degree

    # ── QUEEN-specific per-Gaussian attributes ────────────────────────────────
    # QUEEN's render_mask() accesses `pc.get_flow` (a method returning a
    # per-Gaussian tensor) and `pc.mask_all` (a per-Gaussian update-mask).
    # Both must be indexed to match the proxy's subset.

    @property
    def get_flow(self) -> torch.Tensor:
        """Per-Gaussian optical-flow tensor, indexed to this proxy's subset.

        QUEEN's render_mask() accesses ``pc.get_flow`` (no parentheses), so
        this must be a property — not a plain method — even though the
        underlying GaussianModel exposes it as a regular method.

        Falls back to zeros if the underlying model has no get_flow.
        """
        g_flow = getattr(self._g, "get_flow", None)
        if g_flow is None:
            return torch.zeros(self._n, 3, device=self._idx.device)
        raw = g_flow() if callable(g_flow) else g_flow
        if isinstance(raw, torch.Tensor):
            return self._sel(raw)
        return torch.zeros(self._n, 3, device=self._idx.device)

    @property
    def mask_all(self) -> torch.Tensor:
        """Per-Gaussian update-mask [N, 7], indexed to the proxy subset."""
        raw = getattr(self._g, "mask_all", None)
        if raw is None or not isinstance(raw, torch.Tensor):
            # Return all-ones mask if underlying model has no mask_all
            n_attrs = 7  # QUEEN uses 7-column mask (xyz, f_dc, f_rest, opa, scl, rot, flow)
            return torch.ones(self._n, n_attrs, dtype=torch.bool,
                              device=self._idx.device)
        return self._sel(raw)

    # ── Scalar / model-level passthrough ──────────────────────────────────────
    # Any attribute not explicitly defined on the proxy is forwarded to the
    # underlying model (read-only).  This covers baseline-specific scalars and
    # non-tensor attributes such as QUEEN's frame_idx, latent_decoders,
    # gate_atts, etc.  Tensor attributes whose first dimension equals the full
    # model's Gaussian count are automatically indexed to the proxy subset.

    def __getattr__(self, name: str):
        # __getattr__ is only called when normal attribute lookup fails.
        # Guard against infinite recursion before '_g' is set by __init__.
        if name == "_g":
            raise AttributeError(name)
        val = getattr(self._g, name)
        # Auto-index per-Gaussian tensors (first dim == total Gaussian count)
        if isinstance(val, torch.Tensor) and val.ndim >= 1:
            n_total = self._g.get_xyz.shape[0]
            if val.shape[0] == n_total and n_total > 0:
                return self._sel(val)
        return val

    # ── Utility ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        total = self._g.get_xyz.shape[0]
        pct   = 100.0 * self._n / max(total, 1)
        return (
            f"MaskedGaussianProxy({self._n}/{total} Gaussians, "
            f"{pct:.1f}% of total)"
        )
