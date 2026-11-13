"""Speaker embeddings: whose voice is this (docs/ROBOT_BRAIN_PLAN.md, B4).

`fbank` = Kaldi filterbank features (the model's front end, gated against
Kaldi's own code); `model` = WeSpeaker's ResNet34 trained on VoxCeleb
(CC BY 4.0), a voice as 256 numbers, gated against the published ONNX file.
"""

from .fbank import kaldi_fbank, subtract_mean, pcm_int16_range, FB_MELS, FB_WIN, FB_HOP
from .model import SpeakerNet, cosine, SPK_EMB, SPK_MELS
