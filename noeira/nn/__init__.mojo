"""The storage-passing neural-network framework.

The sole nn framework (the legacy stateless-`Module`/`TensorPack[MutAnyOrigin]`
surface was removed; this storage-passing design was promoted from `nn/storage/`
up to `nn/`). Leaves take owned `Tensor` storages by `ref`/`mut` and build their
typed views INTERNALLY, so the wildcard origin is gone from leaf bodies — the
only residual erasure is the GPU kernel-arg ABI (`MutAnyOrigin`) and the one
load-bearing `TensorPack.__getitem__` pin.

Layout: core / primitives / combinators / optimizer / loss / models / training,
plus the framework-agnostic shared infra (constants / random / datasets /
primitives.ops / core.{element_op, …, target_storage, ptr}).
"""

from .core.tensor import Tensor, TensorImpl
from .core.tensor_refs import TensorRefs
from .core.tensor_pack import TensorPack
from .core.param import Param, ParamVisitor, IsParam
from .core.state import State, IsState
from .core.module import Module
from .core.walkers import for_each_param_auto, zero_grad_auto, join_name
from .core.named_params import NamedParam, named_params, named_states
from .core.describe import describe, print_describe, DescribeVisitor
from .core.initializer import (
    Initializer,
    Kaiming,
    Xavier,
    Zero,
    Normal,
    Deterministic,
)
from .core.checkpoint import save_params, load_params
# safetensors: the INTEROP format. `save_params`/`load_params` above stay the
# resume format — they carry Adam's moments, which safetensors has nowhere to
# put that another framework would understand.
from .core.safetensors_io import (
    SafeTensorsLoader,
    SafeTensorsSaver,
    load_safetensors,
    save_safetensors,
)
from .core.torch_names import (
    LoadTorchNamed,
    SaveTorchNamed,
    TorchNameMap,
    TN_PLAIN,
    TN_TRANSPOSE,
    TN_ZEROS,
)
from .primitives.linear import Linear
from .primitives.linear_relu import LinearReLU
from .primitives.linear_act import LinearAct
from .primitives.linear_tanh import LinearTanh
from .primitives.linear_mish import LinearMish
from .primitives.linear_sigmoid import LinearSigmoid
from .primitives.linear_swish import LinearSwish
from .primitives.block_linear import BlockLinear
from .primitives.tied_linear import TiedLinear
from .primitives.gru_cell import GRUCell
from .primitives.lstm_cell import LSTMCell
from .primitives.lstm_seq import LSTMSeq
from .primitives.attention import ScaledDotProductAttention
from .primitives.masked_attention import (
    MaskedAttention,
    causal_mask,
    all_allow_mask,
    build_modality_mask,
)
from .primitives.add import Add
from .primitives.elementwise import Elementwise
from .primitives.activations import (
    ReLU,
    Tanh,
    Sigmoid,
    GELU,
    GELUTanh,
    Mish,
    Swish,
    Symlog,
)
from .primitives.conv2d import Conv2D
from .primitives.batch_norm_1d import BatchNorm1D
from .primitives.batch_norm_2d import BatchNorm2D
from .primitives.layer_norm import LayerNorm
from .primitives.rms_norm import RMSNorm
from .primitives.min_max_norm import MinMaxNorm
from .primitives.sim_norm import SimNorm
from .primitives.noisy_linear import NoisyLinear
from .primitives.flatten import Flatten
from .primitives.dropout import Dropout
from .primitives.avg_pool_2d import AvgPool2D
from .primitives.max_pool_2d import MaxPool2D
from .primitives.embedding import Embedding
from .primitives.rsample import RSample
from .primitives.binary_elementwise import (
    BinaryElementwise,
    BinaryElemMin,
    BinarySub,
)
from .primitives.concat import Concat, Concat2
from .primitives.scale import Scale
from .primitives.clamp import Clamp
from .primitives.gather_cols import GatherCols
from .primitives.dueling_head import DuelingHead
from .primitives.dueling_head_c51 import DuelingHeadC51

# Phase E remainder — RL-utility + LeWM leaves
from .primitives.slice import Slice
from .primitives.reduce import Reduce, Sum, Mean, ReduceOp, SumOp, MeanOp
from .primitives.reduce_max import ReduceMax
from .primitives.gather_action_slice import GatherActionSlice
from .primitives.zero_linear import ZeroLinear
from .primitives.silu import SiLU
from .primitives.stop_grad import StopGrad
from .primitives.sigreg import SIGReg
from .primitives.layer_norm_no_affine import LayerNormNoAffine
from .primitives.mse_per_sample import MSEPerSample
from .primitives.gate import Gate
from .primitives.modulate import Modulate

# Phase E remainder — transformer / Dreamer4 plumbing leaves
from .primitives.bias_add import BiasAdd
from .primitives.transpose_2d import Transpose2D
from .primitives.token_mean import TokenMean
from .primitives.qkv_to_major import QKVToMajor
from .primitives.swiglu import SwiGLU
from .primitives.space_time_transpose import SpaceTimeTranspose
from .primitives.sinusoidal_pos import SinusoidalPosAdd
from .primitives.sinusoidal_pos_tokens import (
    SinusoidalPos1DTokens,
    SinusoidalPos2DTokens,
    ZeroTokens,
)
from .primitives.cross_attention import (
    CrossAttention,
    SelfAttentionPos,
    SelfAttentionPosMasked,
)
from .primitives.gaussian_vae import GaussianKLStdNormal, GaussianReparam
from .primitives.l1_masked_per_sample import L1MaskedPerSample
from .primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from .primitives.broadcast_tokens import BroadcastTokens
from .primitives.learned_tokens import LearnedTokens
from .primitives.learned_queries import LearnedQueries
from .primitives.mae_replacer import MAEReplacer

# Phase F — transformer/Dreamer4 BLOCKS (composites over models.transformer)
from .primitives.modality_space_attention import ModalitySpaceAttention
from .primitives.dynamics_space_attention import (
    DynamicsSpaceAttention,
    DYN_MOD_ACTION,
    DYN_MOD_SIGNAL,
    DYN_MOD_STEP,
    DYN_MOD_SPATIAL,
    DYN_MOD_REGISTER,
    DYN_MOD_AGENT,
)
from .primitives.decoder_block import DecoderBlock
from .primitives.time_attention_latents import TimeAttentionLatents
from .primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)
from .combinators.sequential import Sequential
from .combinators.residual import Residual
from .combinators.parallel import Parallel
from .combinators.repeat import Repeat
from .combinators.checkpointed import Checkpointed
from .combinators.repeat_conditional import RepeatConditional
from .combinators.projected_residual import ProjectedResidual
from .combinators.tokenwise import Tokenwise
from .combinators.skip_concat import SkipConcat
from .combinators.branch_concat import BranchConcat
from .combinators.stop_grad_params import StopGradParams
from .combinators.compute_graph import ComputeGraph
from .combinators.graph_module2 import GraphModule2, TwoInputGraph
from .combinators.graph_decl import (
    GraphDecl,
    InputSlot,
    Node,
    ExternalNode,
    IsExternal,
)
from .optimizer.optimizer import Optimizer
from .optimizer.param_arena import ParamArena, polyak_arenas
from .optimizer.sgd import SGD
from .optimizer.adam import Adam, AdamW
from .optimizer.schedules import LinearWarmupSchedule
from .optimizer.grad_clip import clip_grad_norm
from .optimizer.scalar_adam import ScalarAdam
from .loss.mse import mse_forward, mse_backward
from .loss.mse_loss import MSELoss
from .loss.cross_entropy import CrossEntropyLoss
from .loss.soft_cross_entropy import SoftCrossEntropyLoss
from .loss.gaussian_nll_loss import GaussianNLLLoss
from .loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from .loss.two_hot import (
    compute_bins,
    compute_symlog_bins,
    fill_bins,
    fill_symlog_bins,
    two_hot_encode,
    two_hot_encode_batch,
    two_hot_encode_symlog_batch,
    decode_value,
    decode_value_batch,
    decode_value_batch_linear,
    symlog,
    symexp,
)
from .training.trainer import Trainer
