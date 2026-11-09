"""Ground-truth 2D occupancy of a MuJoCo scene's height band — the P1 map reference.

A column (cell) is occupied when any of its sample points, a 3 x 3 grid in the
cell times every `dz` between `z_lo` and `z_hi`, lies inside a geom that is
neither the robot's nor invisible. Point-in-geom is computed here from
MuJoCo's own `geom_xpos` / `geom_xmat` / `geom_size`, so the reference shares
nothing with noeira's ray, footprint or map code.

Handles sphere, capsule, ellipsoid, cylinder, box. Planes and heightfields are
the floor (below any sensible band) and are skipped; a MESH outside the robot
raises, because silently skipping it would make it free space in the truth.
"""

import numpy as np


def _inside(gtype, size, p):
    """p: [N, 3] points in the geom frame -> bool [N]."""
    if gtype == 2:  # sphere
        return np.sum(p * p, axis=1) <= size[0] ** 2
    if gtype == 3:  # capsule
        z = np.clip(p[:, 2], -size[1], size[1])
        q = p.copy()
        q[:, 2] -= z
        return np.sum(q * q, axis=1) <= size[0] ** 2
    if gtype == 4:  # ellipsoid
        return np.sum((p / size[None, :3]) ** 2, axis=1) <= 1.0
    if gtype == 5:  # cylinder
        return (np.abs(p[:, 2]) <= size[1]) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= size[0] ** 2)
    if gtype == 6:  # box
        return np.all(np.abs(p) <= size[None, :3], axis=1)
    raise ValueError(f"geom type {gtype} not handled")


def truth_cells(m, d, x0, y0, res, nx, ny, z_lo, z_hi, robot_root, dz=0.025, sub=3, only=None):
    """bool [ny * nx], row j column i — the layout of VoxelMap.occupied_2d.
    `only`: a set of geom ids to restrict to (one object's footprint)."""
    occ = np.zeros((ny, nx), dtype=bool)
    off = (np.arange(sub) + 0.5) / sub * res
    zs = np.arange(z_lo + 0.5 * dz, z_hi, dz)
    for g in range(m.ngeom):
        if only is not None and g not in only:
            continue
        gt = int(m.geom_type[g])
        if gt in (0, 1):
            continue
        b = int(m.geom_bodyid[g])
        if robot_root >= 0 and int(m.body_rootid[b]) == robot_root:
            continue
        matid = int(m.geom_matid[g])
        alpha = m.mat_rgba[matid][3] if matid >= 0 else m.geom_rgba[g][3]
        if alpha == 0:
            continue
        if gt == 7:
            raise ValueError(f"mesh geom {g} outside the robot: not handled by the truth")
        c = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        rb = float(m.geom_rbound[g])
        # Only the cells under the geom's bounding sphere.
        i0 = max(0, int(np.floor((c[0] - rb - x0) / res)))
        i1 = min(nx - 1, int(np.floor((c[0] + rb - x0) / res)))
        j0 = max(0, int(np.floor((c[1] - rb - y0) / res)))
        j1 = min(ny - 1, int(np.floor((c[1] + rb - y0) / res)))
        if i1 < i0 or j1 < j0:
            continue
        ii = np.arange(i0, i1 + 1)
        jj = np.arange(j0, j1 + 1)
        X = (x0 + ii[:, None] * res + off[None, :]).reshape(-1)  # [ni*sub]
        Y = (y0 + jj[:, None] * res + off[None, :]).reshape(-1)
        GX, GY, GZ = np.meshgrid(X, Y, zs, indexing="ij")
        P = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1)
        inside = _inside(gt, m.geom_size[g], (P - c) @ R)
        inside = inside.reshape(len(ii), sub, len(jj), sub, len(zs)).any(axis=(1, 3, 4))
        occ[j0 : j1 + 1, i0 : i1 + 1] |= inside.T
    return occ.reshape(-1)


def geoms_by_prefix(m, prefix):
    """Geom ids whose name starts with `prefix` (the public room names an
    object's parts `table_top`, `table_leg1`, ...)."""
    import mujoco
    out = set()
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if nm and nm.startswith(prefix):
            out.add(g)
    return out
