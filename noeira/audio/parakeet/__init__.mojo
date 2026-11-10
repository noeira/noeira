"""Parakeet-TDT-0.6B-v3 (FastConformer + token-and-duration transducer) in `nn`.

A port of `nvidia/parakeet-tdt-0.6b-v3` (CC-BY-4.0), gated stage by stage
against transformers' `ParakeetForTDT` (`tests/manifests/parakeet-ref.txt`).
See `stt.mojo` for the entry point and docs/NN_MODEL_PORTS_PLAN.md for the plan.
"""

from .stt import ParakeetSTT, PkResult
from .weights import ParakeetTDT
from .tokenizer import ParakeetTokenizer
