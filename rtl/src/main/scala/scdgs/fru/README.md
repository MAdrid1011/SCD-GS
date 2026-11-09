# FRU

`FusedResidencyUnit.scala` implements event-fused residency metadata. It owns
the direct-indexed tier directory, accepts commits, confirmations, and
training-render reads, and emits active, resident, or candidate placement with
a refill indication. Confirmations update window metadata without incrementing
hotness or entering StateSPM.
