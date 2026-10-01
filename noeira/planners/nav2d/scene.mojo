"""Footprints from a parsed MJCF scene — what a ground robot can bump into.

    var fm = parse_model_runtime(scene_path)            # names + geom records
    var skip = List[Bool](length=len(fm.body_names), fill=False)
    for b in robot_bodies: skip[b] = True               # the robot is not an obstacle
    var fp = footprints_from_model(fm, xpos, xquat, skip)

`xpos` / `xquat` are the BODY world poses after forward kinematics, flat,
`xquat` in (x, y, z, w) order — the layout `Phyics3dEnv.get_xpos/get_xquat`
expose. Geom records are body-local (`GeomData.pos_*`, `quat_*`).

Kept: every geom that can collide (contype or conaffinity non-zero), whose
vertical extent overlaps [z_min, z_max], projected onto the ground:

  box      upright -> oriented rectangle; tilted -> bounding circle
  cylinder upright -> circle; tilted -> bounding circle
  sphere   circle
  capsule  upright -> circle; tilted -> circle of radius r + half_length
  plane    skipped (a floor is not an obstacle)
  mesh / ellipsoid / hfield   SKIPPED AND COUNTED — the caller is told how
           many, because a silently missing obstacle is the failure a
           costmap exists to prevent.

⚠ VISUAL-ONLY GEOMS ARE NOT OBSTACLES (contype = conaffinity = 0): the room's
wood floor is a 1 mm box that the robot walks on, and treating it as a
footprint would block the whole room.
"""

from std.math import atan2, sqrt

from noeira.physics3d.parser.flat_model import FlatModelDef
from .costmap import Footprint


comptime _GEOM_PLANE: Int = 0
comptime _GEOM_SPHERE: Int = 1
comptime _GEOM_CAPSULE: Int = 2
comptime _GEOM_BOX: Int = 3
comptime _GEOM_CYLINDER: Int = 4

comptime UPRIGHT_COS: Float64 = 0.98
"""A geom whose local z axis is within ~11 deg of world z counts as upright."""


def _qmul(
    ax: Float64, ay: Float64, az: Float64, aw: Float64,
    bx: Float64, by: Float64, bz: Float64, bw: Float64,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Hamilton product a * b, both (x, y, z, w)."""
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate(
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    vx: Float64, vy: Float64, vz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """v rotated by unit quaternion q (x, y, z, w)."""
    var tx = 2.0 * (qy * vz - qz * vy)
    var ty = 2.0 * (qz * vx - qx * vz)
    var tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


struct SceneFootprints(Movable):
    var footprints: List[Footprint]
    var skipped_unsupported: Int   # mesh / ellipsoid / hfield that could collide

    def __init__(out self):
        self.footprints = List[Footprint]()
        self.skipped_unsupported = 0


def footprint_of_geom(
    geom_type: Int, cx: Float64, cy: Float64, cz: Float64,
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    radius: Float64, half_length: Float64,
    half_x: Float64, half_y: Float64, half_z: Float64,
    z_min: Float64, z_max: Float64,
) -> Tuple[Bool, Footprint]:
    """One WORLD-posed geom -> (kept, footprint). Exposed for tests."""
    var up = _rotate(qx, qy, qz, qw, 0.0, 0.0, 1.0)
    var upright = up[2] >= UPRIGHT_COS or up[2] <= -UPRIGHT_COS
    var none = Footprint.circle(0.0, 0.0, 0.0)
    if geom_type == _GEOM_BOX:
        var ext_z = half_z if upright else sqrt(half_x ** 2 + half_y ** 2 + half_z ** 2)
        if cz + ext_z < z_min or cz - ext_z > z_max:
            return (False, none)
        if upright:
            var ax = _rotate(qx, qy, qz, qw, 1.0, 0.0, 0.0)
            return (True, Footprint.rect(cx, cy, half_x, half_y, atan2(ax[1], ax[0])))
        return (True, Footprint.circle(cx, cy, sqrt(half_x ** 2 + half_y ** 2 + half_z ** 2)))
    if geom_type == _GEOM_SPHERE:
        if cz + radius < z_min or cz - radius > z_max:
            return (False, none)
        return (True, Footprint.circle(cx, cy, radius))
    if geom_type == _GEOM_CYLINDER or geom_type == _GEOM_CAPSULE:
        var hl = half_length
        var ext_z = hl + (radius if geom_type == _GEOM_CAPSULE else 0.0)
        if not upright:
            ext_z = hl + radius
        if cz + ext_z < z_min or cz - ext_z > z_max:
            return (False, none)
        if upright:
            return (True, Footprint.circle(cx, cy, radius))
        return (True, Footprint.circle(cx, cy, radius + hl))
    return (False, none)


def footprints_from_model(
    fm: FlatModelDef, xpos: List[Float64], xquat: List[Float64],
    skip_body: List[Bool], z_min: Float64 = 0.02, z_max: Float64 = 2.0,
) raises -> SceneFootprints:
    """Every colliding, supported geom of every non-skipped body, as a
    ground footprint. See the module docstring for the rules."""
    var out = SceneFootprints()
    for gi in range(len(fm.geoms)):
        ref g = fm.geoms[gi]
        var b = g.body_id
        if b < 0 or b >= len(skip_body):
            raise Error("footprints_from_model: geom " + String(gi) + " has body " + String(b))
        if skip_body[b]:
            continue
        if g.contype == 0 and g.conaffinity == 0:
            continue
        if g.geom_type == _GEOM_PLANE:
            continue
        if (g.geom_type != _GEOM_BOX and g.geom_type != _GEOM_SPHERE
                and g.geom_type != _GEOM_CYLINDER and g.geom_type != _GEOM_CAPSULE):
            out.skipped_unsupported += 1
            continue
        var bx = xquat[b * 4 + 0]
        var by = xquat[b * 4 + 1]
        var bz = xquat[b * 4 + 2]
        var bw = xquat[b * 4 + 3]
        var off = _rotate(bx, by, bz, bw, g.pos_x, g.pos_y, g.pos_z)
        var q = _qmul(bx, by, bz, bw, g.quat_x, g.quat_y, g.quat_z, g.quat_w)
        var r = footprint_of_geom(
            g.geom_type,
            xpos[b * 3 + 0] + off[0], xpos[b * 3 + 1] + off[1], xpos[b * 3 + 2] + off[2],
            q[0], q[1], q[2], q[3],
            g.radius, g.half_length, g.half_x, g.half_y, g.half_z, z_min, z_max,
        )
        if r[0]:
            out.footprints.append(r[1])
    return out^
