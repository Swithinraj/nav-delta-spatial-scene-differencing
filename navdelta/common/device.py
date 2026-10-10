"""Compute device for the model stages: CUDA in fp16 when available, otherwise CPU in fp32.

Set NAVDELTA_DEVICE=cpu to force the CPU. Results are deterministic for a given
device, but fp16 on a GPU and fp32 on a CPU can differ in the last digits of a
score, which can occasionally flip a close label decision.
"""

from __future__ import annotations

import os

import torch

if os.environ.get("NAVDELTA_DEVICE", "").lower() == "cpu" or not torch.cuda.is_available():
    DEVICE, DTYPE = "cpu", torch.float32
else:
    DEVICE, DTYPE = "cuda", torch.float16

PRECISION = "fp16 on CUDA" if DEVICE == "cuda" else "fp32 on CPU"
