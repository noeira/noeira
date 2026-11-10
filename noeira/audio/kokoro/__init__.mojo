"""Kokoro-82M (StyleTTS 2 + iSTFTNet) text-to-speech in Mojo.

A port of `hexgrad/Kokoro-82M` (Apache-2.0), gated stage by stage against
hexgrad's own `KModel` (`tests/manifests/kokoro-ref.txt`). See `tts.mojo` for
the entry point (text or phonemes in, 24 kHz speech out; CPU), `g2p.mojo` for
text -> phonemes through espeak-ng (`noeira/audio/espeak.mojo`), and
docs/NN_MODEL_PORTS_PLAN.md §11 for the architecture and what is gated how.
"""

from .tts import KokoroTTS, KokoroSpeech
from .g2p import KK_G2P_DEPLOYMENT, KK_G2P_MISAKI, KK_G2P_MISAKI_EN
from .weights import KokoroWeights
