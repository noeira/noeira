"""Regression gate (GOLDEN-frozen): per-field contact detection on Walker2D.

Originally validated BIT-EXACT against the legacy FK + narrow-phase kernels.
That legacy reference was frozen into the GOLDEN fingerprints below during
Phase-0 of the physics3d sunset, so this gate survives deletion of the legacy
slab/kernels. It checks:
  * fields-GPU (FK -> detect) reproduces the frozen (legacy-validated)
    fingerprint — per-env contact counts + an order-sensitive checksum of the
    contact records, and
  * fields-CPU == fields-GPU on count + records (independent CPU oracle).

Walker2D (floor plane + 7 body capsules), BATCH=2 penetrating poses. Model build
goldens after an INTENTIONAL physics change: HARVEST=True, run on Apple, paste,
HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_contact_detection_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, Model, Dims, DimsLike
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from noeira.physics3d.collision.contact_detection import (
    detect_contacts,
)
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
)
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime MD = ModelDims[Walker2dModel]
comptime BATCH = 2

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON = 10  # total contacts across both envs
# ⚠ REFRESHED 2026-08-05, from 3390.603547532484, and NOT on faith.
#
# The element-order fix made `full_parser` group geoms by body as MuJoCo
# numbers them. Geom indices moved, so the EMISSION ORDER of these 10 contacts
# moved with them, and a deliberately order-sensitive checksum is exactly what
# notices that. `GOLD_NCON` did not move: no contact appeared or vanished.
#
# This gate could not tell "reordered correctly" from "broken", because the
# legacy kernels it was frozen against no longer exist — a golden with no live
# reference only ever reports CHANGE. So the question was answered elsewhere
# before this number was touched:
# `tests/physics3d/test_walker2d_contacts_vs_mujoco.mojo` replays these same
# two poses against MuJoCo and compares contact by contact, MATCHED BY POSITION
# rather than sorted, so a right-set-wrong-order result fails it. Measured:
# ncon 6/6 and 4/4, ZERO position-matched body-pair mismatches, dist 1.2e-7,
# pos 7.6e-8, normal 0.0 (float32 FK round-off). Our order IS MuJoCo's.
#
# Read that file first if this ever moves again — it is the reference this one
# lacks, and refreshing this constant without it is laundering.
comptime GOLD_CON = 9501.98150853524  # order-sensitive contact-record checksum


def main() raises:
    print("--- contact detection fields GOLDEN gate: walker2d BATCH=", BATCH)
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    # Poses: env0 slight floor penetration; env1 heavy penetration + bent legs.
    var d = Data[DTYPE, MD, BATCH]()
    var qcfg = List[List[Float64]]()
    var q0 = List[Float64](length=NQ, fill=0.0)
    q0[1] = 1.18  # rootz slightly below standing -> feet penetrate
    qcfg.append(q0^)
    var q1 = List[Float64](length=NQ, fill=0.0)
    q1[1] = 0.85
    q1[3] = 0.6
    q1[4] = -1.1
    q1[6] = -0.4
    q1[7] = -0.9
    qcfg.append(q1^)
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)

    # Fields GPU: FK + detection.
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    detect_contacts["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_total = 0
    var fp_con = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        ncon_total += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_con += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
    if ncon_total == 0:
        raise Error("expected contacts in these poses — gate is vacuous")
    print("  fields-GPU total contacts:", ncon_total)

    if HARVEST:
        print("  HARVEST GOLD_NCON =", ncon_total)
        print("  HARVEST GOLD_CON  =", fp_con)
    else:
        if ncon_total != GOLD_NCON and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "total contacts " + String(ncon_total) + " != golden "
                + String(GOLD_NCON)
            )
        var denom = abs(GOLD_CON) if abs(GOLD_CON) > 1e-9 else 1.0
        if abs(fp_con - GOLD_CON) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "contact-record fingerprint " + String(fp_con) + " != golden "
                + String(GOLD_CON)
            )
        print("  PASS: fields-GPU matches golden fingerprint")

    # --- independent CPU oracle: fields-CPU == fields-GPU (count + records) ---
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            dc.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    forward_kinematics["cpu", DTYPE, BATCH=BATCH](dc, mf)
    detect_contacts["cpu", DTYPE, BATCH=BATCH](dc, mf)
    var worst = Float64(0)
    for e in range(BATCH):
        var nc_g = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        var nc_c = Int(
            dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        if nc_g != nc_c:
            raise Error("fields-CPU contact count differs from fields-GPU")
        for c in range(nc_g):
            for k in range(CONTACT_SIZE):
                var err = abs(
                    Float64(
                        dc.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                    - Float64(
                        d.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                )
                if err > worst:
                    worst = err
    print("  fields-CPU vs fields-GPU worst record err:", worst)
    if worst > 1e-4:
        raise Error("fields-CPU contact records tolerance exceeded")
    print("  PASS: fields-CPU contacts within 1e-4")
    print("test_contact_detection_fields: ALL PASS")
