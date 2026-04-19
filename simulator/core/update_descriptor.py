"""
UpdateDescriptor — unified representation of one per-Gaussian update.
UpdateStream    — collection of descriptors for one frame, with traffic stats.
AttrClass       — canonical attribute taxonomy shared by all 4 models.

Design note
-----------
This module is **fully model-agnostic**. It knows nothing about 3DGStream,
QUEEN, HiCoM, or Swift4D. The only model-specific knowledge lives in the
Operator Slots (operators/op_*.py), which translate each model's optimizer
group names into AttrClass via from_group_name().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Attribute taxonomy ────────────────────────────────────────────────────────

class AttrClass(str, Enum):
    """Canonical attribute categories for Gaussian state.

    The 'NETWORK' category covers non-per-Gaussian parameters: NTC weights,
    latent decoders, gating networks, hash grids, deformation MLPs, etc.
    These receive a different (harder) consolidation threshold.
    """
    XYZ      = "xyz"
    ROTATION = "rotation"
    SCALING  = "scaling"
    OPACITY  = "opacity"
    FEATURES = "features"   # SH coefficients (DC + rest combined)
    NETWORK  = "network"    # non-per-Gaussian network parameters

    @classmethod
    def from_group_name(cls, name: str) -> "AttrClass":
        """Infer AttrClass from an optimizer param-group name string.

        Handles all naming conventions across the four baselines:
          3DGStream : "xyz", "f_dc", "f_rest", "opacity", "scaling",
                      "rotation", "added_xyz", "added_f_dc", ...
          QUEEN     : same Gaussian attrs + "decoder", "gate"
          HiCoM     : same Gaussian attrs + "deform", "deform_optimizer"
          Swift4D   : same Gaussian attrs + "hash" (hash-grid dynamic branch)
        """
        n = name.lower().strip()
        if "xyz" in n:
            return cls.XYZ
        if "rot" in n:
            return cls.ROTATION
        if "scal" in n:
            return cls.SCALING
        if "opac" in n:
            return cls.OPACITY
        if any(tok in n for tok in ("f_dc", "f_rest", "feat", "sh_", "_sh")):
            return cls.FEATURES
        # Fall-through: decoder, gate, ntc, hash, deform, etc.
        return cls.NETWORK

    def payload_floats(self) -> int:
        """Approximate number of float32 values per Gaussian for this attr.

        Used to estimate ASIC update traffic in bytes.
        FEATURES is SH degree-3 = 48 coefficients (DC 3 + rest 45).
        NETWORK is counted as 1 (per-parameter scalar; size varies widely).
        """
        sizes = {
            AttrClass.XYZ:      3,
            AttrClass.ROTATION: 4,
            AttrClass.SCALING:  3,
            AttrClass.OPACITY:  1,
            AttrClass.FEATURES: 48,
            AttrClass.NETWORK:  1,
        }
        return sizes[self]

    def delta_bytes(self) -> int:
        """Byte cost of one Gaussian's delta for this attribute (float32)."""
        return self.payload_floats() * 4


# ── Update Descriptor ─────────────────────────────────────────────────────────

@dataclass
class UpdateDescriptor:
    """Represents a single per-Gaussian (or per-network-parameter) update.

    Produced by GLASSPipeline._capture_stream() from optimizer.param_groups,
    and consumed by ConsolidationUnit.compute_scales().

    Fields
    ------
    target_id : int
        Gaussian index in [0, N). -1 for network-level descriptors.
    attr_class : AttrClass
        Which attribute group this descriptor belongs to.
    delta_norm : float
        L2 norm of the gradient for this (Gaussian, attr) pair.
        Used as a scalar proxy for update magnitude.
    version : int
        Monotonic frame counter; set by GLASSPipeline.begin_frame().
    struct_changed : bool
        True when the Gaussian was spawned, split, or pruned this frame.
        Causes SalienceFirstIssue to classify it FULL (history invalidated).
    grad_scale : float
        Set by ConsolidationUnit after similarity analysis.
        GLASSPipeline reads this value to scale param.grad in-place.
        Default 1.0 = no change (pass-through in Mode 0).
    """
    target_id:      int
    attr_class:     AttrClass
    delta_norm:     float
    version:        int
    struct_changed: bool  = False
    grad_scale:     float = 1.0

    def delta_bytes(self) -> int:
        """Byte cost of transmitting this descriptor's payload."""
        return self.attr_class.delta_bytes()


# ── Update Stream ─────────────────────────────────────────────────────────────

class UpdateStream:
    """Accumulates UpdateDescriptors produced during one frame.

    After consolidation, grad_scale fields are filled in. The stream then
    provides traffic-accounting statistics used in the ablation JSON output.
    """

    def __init__(self) -> None:
        self.descriptors: list[UpdateDescriptor] = []

    def add(self, d: UpdateDescriptor) -> None:
        self.descriptors.append(d)

    # ── Traffic statistics ────────────────────────────────────────────────────

    def total_bytes(self) -> int:
        """Total update traffic bytes for this frame (before suppression)."""
        return sum(d.delta_bytes() for d in self.descriptors)

    def suppressed_bytes(self) -> int:
        """Bytes saved by consolidation (descriptors with grad_scale < 1.0)."""
        return sum(
            d.delta_bytes()
            for d in self.descriptors
            if d.grad_scale < 1.0
        )

    def active_bytes(self) -> int:
        """Bytes that actually propagate (total - suppressed)."""
        return self.total_bytes() - self.suppressed_bytes()

    def suppression_ratio(self) -> float:
        """Fraction of bytes suppressed. Returns 0.0 on empty stream."""
        total = self.total_bytes()
        if total == 0:
            return 0.0
        return self.suppressed_bytes() / total

    def unique_targets(self) -> set[int]:
        """Set of distinct Gaussian indices (target_id >= 0) in this stream."""
        return {d.target_id for d in self.descriptors if d.target_id >= 0}

    def struct_changed_ids(self) -> list[int]:
        """List of Gaussian IDs that had structural changes this frame."""
        return [d.target_id for d in self.descriptors
                if d.struct_changed and d.target_id >= 0]

    def __len__(self) -> int:
        return len(self.descriptors)

    def __repr__(self) -> str:
        return (
            f"UpdateStream(n={len(self)}, "
            f"total_kb={self.total_bytes()/1024:.1f}, "
            f"suppressed_ratio={self.suppression_ratio():.2%})"
        )
