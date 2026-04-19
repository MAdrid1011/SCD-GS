"""
GLASS ASIC Simulator
====================
A Python simulator for the GLASS (Live-State Dataflow Architecture for
Streamable Gaussian System) ASIC.

Architecture
------------
- GLASSPipeline  : shared Stages 1-5, model-agnostic
- BaseOperatorSlot: per-model Stage-0 hooks (thin translation layer)
- DatasetConfig  : per-dataset paths + parameters

Usage
-----
    from simulator.config import ASICConfig
    from simulator.pipeline import GLASSPipeline
    from simulator.operators.op_3dgstream import Op3DGStream
    from simulator.datasets import load_dataset_config

    ds  = load_dataset_config("flame_steak")
    cfg = ASICConfig(mode=3)
    slot= Op3DGStream()
    pipeline = GLASSPipeline(cfg, slot)
"""

from simulator.config import ASICConfig, AblationMode

__all__ = ["ASICConfig", "AblationMode"]
