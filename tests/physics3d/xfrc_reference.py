"""MuJoCo's answers for tests/physics3d/test_xfrc_vs_mujoco.mojo, written between its `xfrc-reference` markers.

    pixi run python -I tests/physics3d/xfrc_reference.py        (from the repo root)

The fixtures (XML strings, wrench tables, step counts) are READ FROM THE TEST FILE, so the two sides step the same thing.
`xfrc_applied` is MuJoCo's own field; the `chain_point` case goes through `mj_applyFT` into `qfrc_applied` instead, a
second path for a force at a point.
"""
import re, sys
import numpy as np, mujoco

TEST = "tests/physics3d/test_xfrc_vs_mujoco.mojo"
src = open(TEST).read()
xml = {k: v for k, v in re.findall(r'comptime (XML_\w+) = String\(\s*"""(.*?)"""\s*\)', src, re.S)}
g1_path = re.search(r'comptime G1_XML = String\("([^"]+)"\)', src).group(1)
const = {k: float(v) for k, v in re.findall(r"comptime (\w+_STEPS_[AB]|G1_AIR_DZ) = ([0-9.]+)", src)}
wblock = src[src.index("# <xfrc-wrenches>"): src.index("# </xfrc-wrenches>")]
tables = {}
for name, body in re.findall(r"def (_w_\w+)\(\) -> List\[\w+\]:\n(.*?)(?=\n\n\ndef |\Z)", wblock, re.S):
    body = re.sub(r'""".*?"""', "", body, flags=re.S)
    tables[name] = eval(body[body.index("return") + 6:].strip().rstrip("#").strip())
assert set(xml) == {"XML_CHAIN", "XML_BOX"} and len(tables) == 5, (list(xml), list(tables))


def check(m):
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL


def roll(m, dz, rows, a, b, via_applyft=False, need_contact=False):
    """rows: [(body, kind, six numbers)]; (qpos, qvel) after a steps then after b."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] += dz
    k = mujoco.MjData(m)                      # kinematics of the start state, on a scratch copy
    k.qpos[:] = d.qpos
    mujoco.mj_kinematics(m, k); mujoco.mj_comPos(m, k)
    for body, kind, v in rows:
        if kind == 0:
            d.xfrc_applied[body] = v
        elif via_applyft:
            q = np.zeros(m.nv)
            mujoco.mj_applyFT(m, k, np.array(v[:3]), np.zeros(3), np.array(v[3:]), body, q)
            d.qfrc_applied[:] += q
        else:
            f, p = np.array(v[:3]), np.array(v[3:])
            d.xfrc_applied[body, :3] = f
            d.xfrc_applied[body, 3:] = np.cross(p - k.xipos[body], f)
    out, ncon = [], []
    for t in range(1, int(b) + 1):
        mujoco.mj_step(m, d)
        ncon.append(d.ncon)
        if t == int(a) or t == int(b):
            out += list(d.qpos) + list(d.qvel)
    if need_contact:
        assert min(ncon) > 0, ncon
    return out, ncon


def rows_of(flat):
    return [(int(flat[i]), int(flat[i + 1]), flat[i + 2: i + 8]) for i in range(0, len(flat), 8)]


ref = {}
mc = mujoco.MjModel.from_xml_string(xml["XML_CHAIN"]); check(mc)
A, B = const["CHAIN_STEPS_A"], const["CHAIN_STEPS_B"]
ref["chain_zero"], _ = roll(mc, 0, [], A, B)
ref["chain_wrench"], _ = roll(mc, 0, rows_of(tables["_w_chain"]), A, B)
ref["chain_point"], _ = roll(mc, 0, rows_of(tables["_w_chain_point"]), 1, 1, via_applyft=True)
mb = mujoco.MjModel.from_xml_string(xml["XML_BOX"]); check(mb)
A, B = const["BOX_STEPS_A"], const["BOX_STEPS_B"]
ref["box_zero"], n0 = roll(mb, 0, [], A, B, need_contact=True)
ref["box_wrench"], n1 = roll(mb, 0, rows_of(tables["_w_box"]), A, B, need_contact=True)
print("box contacts per step:", sorted(set(n0)), sorted(set(n1)), file=sys.stderr)

mg = mujoco.MjModel.from_xml_path(g1_path); check(mg)
assert mg.nq == 36 and mg.nv == 35
bid = [mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_BODY, n) for n in tables["_w_g1_names"]]
assert min(bid) > 0
k = mujoco.MjData(mg); mujoco.mj_resetData(mg, k); mujoco.mj_forward(mg, k)
pts = [k.geom_xpos[mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_GEOM, s + "_hand_collision")].copy() for s in ("left", "right")]
F = tables["_w_g1_forces"]


def g1_rows(dz):
    r = []
    for i in range(2):
        r.append((bid[i], 1, F[i * 6: i * 6 + 3] + [pts[i][0], pts[i][1], pts[i][2] + dz]))
    r.append((bid[2], 0, F[12:18]))
    return r


# the floor leg's height: the first drop (0.5 mm steps) at which a contact is 1 mm deep
floor_dz = None
for i in range(0, 800):
    dz = 0.1 - 0.0005 * i
    k = mujoco.MjData(mg); mujoco.mj_resetData(mg, k); k.qpos[2] += dz; mujoco.mj_forward(mg, k)
    if k.ncon >= 2 and min(c.dist for c in k.contact[: k.ncon]) <= -0.001:
        floor_dz = dz
        print("G1 floor leg: dz", dz, "contacts", k.ncon, "deepest", min(c.dist for c in k.contact[: k.ncon]), file=sys.stderr)
        break
assert floor_dz is not None
A, B = const["G1_STEPS_A"], const["G1_STEPS_B"]
air = const["G1_AIR_DZ"]
ref["g1air_zero"], n = roll(mg, air, [], A, B); assert max(n) == 0, n
ref["g1air_wrench"], n = roll(mg, air, g1_rows(air), A, B); assert max(n) == 0, n
ref["g1floor_zero"], n2 = roll(mg, floor_dz, [], A, B, need_contact=True)
ref["g1floor_wrench"], n3 = roll(mg, floor_dz, g1_rows(floor_dz), A, B, need_contact=True)
print("G1 floor contacts per step:", n2, n3, "| body ids", bid, file=sys.stderr)


def lit(name, vals, doc):
    body = ",\n        ".join(", ".join(repr(float(x)) for x in vals[i: i + 4]) for i in range(0, len(vals), 4))
    return f'def {name}() -> List[Float64]:\n    """{doc}"""\n    return [\n        {body},\n    ]\n'


parts = [f"# MuJoCo {mujoco.__version__}, written by tests/physics3d/xfrc_reference.py. Do not edit by hand.\n"]
for kname, vals in ref.items():
    parts.append(lit("_ref_" + kname, vals, "(qpos, qvel) after the first step count, then after the second."))
parts.append(lit("_g1_points", list(pts[0]) + list(pts[1]), "The hand capsules' centres at qpos0: left, right."))
parts.append(f'def _g1_floor_dz() -> Float64:\n    """The root\'s drop from qpos0 at which a foot contact is 1 mm deep."""\n    return {floor_dz!r}\n')
block = "# <xfrc-reference>\n" + "\n\n".join(parts) + "# </xfrc-reference>"
new = src[: src.index("# <xfrc-reference>")] + block + src[src.index("# </xfrc-reference>") + len("# </xfrc-reference>"):]
open(TEST, "w").write(new)
print("wrote", sum(len(v) for v in ref.values()), "reference values into", TEST, file=sys.stderr)
