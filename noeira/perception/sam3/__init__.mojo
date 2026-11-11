"""SAM 3 (Meta, `facebook/sam3`, the SAM licence) in Mojo: promptable concept
segmentation — a picture and a noun phrase in, every instance's mask, box and
score out. Ported stage by stage against transformers' `Sam3Model`
(`tools/perception/dump_sam3_reference.py`; docs/NN_MODEL_PORTS_PLAN.md, SAM 3).

    vit.mojo   `Sam3ViT`: the vision backbone (ViT-L/14 at 1008, windowed + global, 2-D RoPE)
    ops.mojo   weights on the device, the GEMM, the shared kernels
"""

from .ops import Sam3Weights, sam3_checkpoint
from .vit import Sam3ViT
