"""A ray-ready copy of a scene for the sensors, beside whatever model steps it.

    var ss = SensorScene("path/to/scene.xml")
    ss.set_qpos(qpos)                  # the simulator's state, copied in
    lidar_scan_cpu[DType.float64, DynDims, 1](ss.d, ss.m, spec, mount, ...)

WHY A SECOND MODEL. A simulator's model is built for the SOLVER: it often
carries no triangle soup (`nmesh_tri = 0`), so its meshes — the robot's whole
visual body — are invisible to rays, and a sensor on it would never see its
own arms. This model is parsed from the same file with the soup sized to what
the meshes need (the loader reports the number when it is short), so rays see
exactly what MuJoCo's `mj_ray` sees. Copying `qpos` and running forward
kinematics is all a scan needs; nothing is stepped here.

⚠ THE TWO MODELS MUST SHARE `qpos` LAYOUT — the same scene file guarantees it;
`set_qpos` checks the length.
"""

from noeira.physics3d.fields import Data, Model, DynDims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    read_model_source,
    dims_from_flat,
    build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.gpu.constants import MODEL_GEOM_SIZE, GEOM_IDX_BODY

comptime _F64 = DType.float64


def ray_ready_dims(path: String) raises -> DynDims:
    """The dims of `path` with the mesh vertex and triangle budgets the loader
    asks for (it raises with the triangle count; vertices are doubled)."""
    var src = read_model_source(path)
    var fmd = parse_xml_full(src[0], src[1])
    var verts = 65536
    var tris = 1
    for _ in range(30):
        var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=verts, nmesh_tri=tris)
        var m = Model[_F64, DynDims](dims)
        try:
            build_model_runtime[_F64](fmd, dims, m)
            return dims
        except e:
            var msg = String(e)
            var k = msg.find("Pass nmesh_tri=")
            if k >= 0:
                var bytes = msg.as_bytes()
                var j = k + 15
                var v = 0
                while j < len(bytes) and bytes[j] >= 48 and bytes[j] <= 57:
                    v = v * 10 + Int(bytes[j]) - 48
                    j += 1
                tris = v
            elif msg.find("mesh vertex capacity") >= 0:
                verts *= 2
            else:
                raise e^
    raise Error("ray_ready_dims: could not size the mesh budgets of " + path)


struct SensorScene(Movable):
    var m: Model[_F64, DynDims]
    var d: Data[_F64, DynDims, 1]
    var dims: DynDims

    def __init__(out self, path: String) raises:
        self.dims = ray_ready_dims(path)
        var src = read_model_source(path)
        var fmd = parse_xml_full(src[0], src[1])
        self.m = Model[_F64, DynDims](self.dims)
        build_model_runtime[_F64](fmd, self.dims, self.m)
        self.d = Data[_F64, DynDims, 1](self.dims)
        # The heightfield grids live in `Data` and start at zero.
        init_hfield_data(self.d, self.m)

    def set_qpos(mut self, q: List[Float64]) raises:
        if len(q) != self.dims.get_nq():
            raise Error("SensorScene.set_qpos: " + String(len(q)) + " values for nq = "
                        + String(self.dims.get_nq()))
        for k in range(len(q)):
            self.d.qpos.data[k] = q[k]
        forward_kinematics["cpu", _F64, DynDims, 1](self.d, self.m)

    def geom_body(self, g: Int) -> Int:
        return Int(self.m.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
