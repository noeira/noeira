"""Audio — the signal processing around the speech models.

Its own package, like `noeira/vision/`: a mel front end or an inverse STFT has no
parameters and is not an `nn` module, and the WAV container stays in
`noeira/io/`. docs/NN_MODEL_PORTS_PLAN.md (gap G4) is the plan this is built to.
"""

from .mel import LogMel, slaney_mel_filterbank, hann_window
from .lfm3_mel import Lfm3LogMel, hf_mel_filter_bank
from .espeak import EspeakNG
