"""Kokoro-82M (StyleTTS 2 + iSTFTNet) text-to-speech in Mojo.

A port of `hexgrad/Kokoro-82M` (Apache-2.0), gated stage by stage against
hexgrad's own `KModel` (`tests/manifests/kokoro-ref.txt`). See `tts.mojo` for
the entry point (phonemes in, 24 kHz speech out; CPU) and
docs/NN_MODEL_PORTS_PLAN.md §11 for the architecture, what is gated and how,
and what is next (the espeak-ng binding for text input, then the device path).
"""

from .tts import KokoroTTS, KokoroSpeech
from .weights import KokoroWeights
