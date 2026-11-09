"""Stage-II event-fused residency oracle with a five-frame metadata window."""

from __future__ import annotations

from math import floor

from .contracts import Tier, TierEntry


class FusedResidencyModel:
    """Models direct-indexed h/d lookup, allocation, and short-window energy."""

    def __init__(self, active_capacity: int = 2, resident_capacity: int = 2, window: int = 5) -> None:
        self.active_capacity = active_capacity
        self.resident_capacity = resident_capacity
        self.window = window
        self.entries: dict[int, TierEntry] = {}
        self.frame_count = 0
        self.active_hits = 0
        self.resident_hits = 0
        self.candidate_hits = 0
        self.refills = 0

    @staticmethod
    def _quantize(strength: float) -> int:
        bounded = max(0.0, min(1.0, strength))
        return max(0, min(255, floor(255.0 * bounded)))

    def _lookup(self, gid: int) -> TierEntry | None:
        entry = self.entries.get(gid)
        if entry is None:
            return None
        if entry.tier is Tier.ACTIVE:
            self.active_hits += 1
        elif entry.tier is Tier.RESIDENT:
            self.resident_hits += 1
        else:
            self.candidate_hits += 1
        return entry

    def candidate_request_accepted(self, gid: int) -> None:
        """Record a cold candidate request only after its line transaction fires."""

        entry = self.entries.get(gid)
        if entry is None or entry.tier is not Tier.CANDIDATE:
            raise ValueError("candidate refill requires a candidate directory entry")
        self.refills += 1

    def commit(self, gid: int, effective_strength: float) -> TierEntry:
        """Apply a state-changing commit and its hotness observation."""

        entry = self._lookup(gid)
        if entry is None:
            entry = TierEntry(gid=gid, tier=Tier.CANDIDATE, row=gid, h=1, d=self._quantize(effective_strength), touched=True)
            self.entries[gid] = entry
        else:
            entry.h = min(15, entry.h + 1)
            entry.d = max(entry.d, self._quantize(effective_strength))
            entry.touched = True
        entry.energy_sum += effective_strength
        entry.samples = min(self.window, entry.samples + 1)
        return entry

    def confirmation(self, gid: int) -> TierEntry:
        """Apply a zero-strength metadata observation without changing h."""

        entry = self.entries.get(gid)
        if entry is None:
            entry = TierEntry(gid=gid, tier=Tier.CANDIDATE, row=gid)
            self.entries[gid] = entry
        entry.d = max(entry.d, self._quantize(0.0))
        entry.samples = min(self.window, entry.samples + 1)
        entry.touched = True
        return entry

    def render_read(self, gid: int) -> TierEntry:
        entry = self._lookup(gid)
        if entry is None:
            entry = TierEntry(gid=gid, tier=Tier.CANDIDATE, row=gid)
            self.entries[gid] = entry
        entry.h = min(15, entry.h + 1)
        entry.touched = True
        return entry

    def finish_frame(self) -> None:
        self.frame_count += 1
        if self.frame_count % self.window:
            return
        ordered = sorted(self.entries.values(), key=lambda entry: (entry.h, entry.d, -entry.gid), reverse=True)
        for rank, entry in enumerate(ordered):
            if rank < self.active_capacity:
                entry.tier = Tier.ACTIVE
                entry.row = rank
            elif rank < self.active_capacity + self.resident_capacity:
                entry.tier = Tier.RESIDENT
                entry.row = rank - self.active_capacity
            else:
                entry.tier = Tier.CANDIDATE
                entry.row = entry.gid
            if not entry.touched:
                entry.h >>= 1
            entry.d = 0
            entry.energy_sum = 0.0
            entry.samples = 0
            entry.touched = False
