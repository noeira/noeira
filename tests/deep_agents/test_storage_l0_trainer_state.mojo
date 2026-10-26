"""L0 gate — storage TrainerState alloc + read/write round-trip (CPU + GPU).

Confirms the migrated `TrainerState` (Scratch -> owned storage Tensor) allocates
the minibatch six-pack on the chosen target and that the buffers are usable:
CPU writes/reads `.data`; GPU uploads/downloads through the Tensor surface.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l0_trainer_state.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l0_trainer_state.mojo
"""

from std.testing import assert_true, assert_equal
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.training.trainer_block import TrainerState


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 8


def test_cpu() raises:
    print("TrainerState CPU alloc + round-trip ...")
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    assert_equal(len(st.mb_s.data), BATCH * OBS, "mb_s sized")
    assert_equal(len(st.mb_a.data), BATCH * ACT, "mb_a sized")
    assert_equal(len(st.mb_r.data), BATCH, "mb_r sized")
    assert_equal(len(st.mb_y.data), BATCH, "mb_y sized")
    for i in range(BATCH * OBS):
        st.mb_s.data[i] = Scalar[DT](i) * 0.5
    var ok = True
    for i in range(BATCH * OBS):
        if st.mb_s.data[i] != Scalar[DT](i) * 0.5:
            ok = False
    assert_true(ok, "mb_s write/read")
    assert_true(not st.has_per, "has_per default false")
    assert_true(st.did_step, "did_step default true")
    print("  ok")


def test_gpu() raises:
    print("TrainerState GPU alloc + upload/download ...")
    var c = DeviceContext()
    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](Optional(c))
    assert_true(Bool(st.mb_s.dev), "mb_s device buffer")
    assert_true(Bool(st.mb_a.dev), "mb_a device buffer")
    # write host -> upload -> zero host -> download -> verify
    st.mb_r.data = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    for b in range(BATCH):
        st.mb_r.data[b] = Scalar[DT](b) - 2.0
    st.mb_r.n = BATCH
    st.mb_r.upload(c)
    for b in range(BATCH):
        st.mb_r.data[b] = Scalar[DT](999)
    st.mb_r.download(c)
    var ok = True
    for b in range(BATCH):
        if st.mb_r.data[b] != Scalar[DT](b) - 2.0:
            ok = False
    assert_true(ok, "mb_r gpu upload/download round-trip")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L0 storage TrainerState gate")
    print("=" * 60)
    test_cpu()
    comptime if has_accelerator():
        test_gpu()
    else:
        print(
            "TrainerState GPU: SKIPPED (no accelerator;"
            " run with -e apple / -e nvidia)"
        )
    print("ALL PASSED")
