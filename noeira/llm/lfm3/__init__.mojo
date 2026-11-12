"""LFM3 (Liquid AI): the text backbone shared by d1, LFM3 text, VL and omni,
LFM3-VL's vision tower (pictures in), LFM3-Omni's audio encoder (clips in), and
generation on a cache (LFM3 as a chat model).
"""

from .model import Lfm3Text, Lfm3Config, LT_CONV, LT_FULL, LT_SLIDING
from .vision import Lfm3Vision, Lfm3VisionConfig
from .audio import Lfm3Audio, Lfm3AudioConfig
from .generate import Lfm3Cache, forward_cached, last_logits
from .chat import Lfm3Chat, ChatSettings, choose
from noeira.ai.picture import Picture
