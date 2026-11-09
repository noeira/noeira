"""Perception in simulation: sensors as data, mounted on any body of any model.

The point of this package is to choose hardware with the simulator before
buying it, so nothing here is written for one sensor or one robot:

  lidar.mojo        `LidarSpec` (a scan pattern + range window + noise, read
                    from a text file — see `zoo/`), `LidarMount` (a pose on
                    any body), the sensor-frame directions, the noise model
  depth.mojo        `DepthCameraSpec` (pinhole intrinsics + ideal / stereo / ToF
                    error model, `zoo/*.depth`), pixel directions, unprojection
  depth_scan.mojo   the depth image: `depth_scan_cpu`, `BatchedDepthCamera`,
                    a mount from an MJCF `<camera>`
  spec_text.mojo    the `key = value` reader every spec file uses
  sensor_scene.mojo `SensorScene`: a ray-ready copy of a scene (triangle soup
                    sized automatically) that a scan reads beside the simulator
  occupancy.mojo    `VoxelMap`: a log-odds voxel band from scans / depth images,
                    projected to the 2D obstacle cells `nav2d.NavGrid` takes
  lidar_scan.mojo   the scan itself over `Model`/`Data`: `lidar_scan_cpu` and
                    `BatchedLidar` (one GPU thread per lane x ray), both on
                    `physics3d.ray.ray_model` (= MuJoCo's `mj_ray`)

Gated against MuJoCo's own `mj_ray` (tests/perception/).
"""

from .lidar import (
    LidarSpec,
    LidarMount,
    LIDAR_FAN,
    LIDAR_RINGS,
    LIDAR_SPIRAL,
    LIDAR_NO_RETURN,
    lidar_spec_from_text,
    load_lidar_spec,
    lidar_directions,
    lidar_uniforms,
    lidar_return,
)
from .lidar_scan import (
    lidar_cast,
    lidar_mount_world,
    lidar_scan_cpu,
    BatchedLidar,
)
from .lidar_scan import mount_world_pose
from .depth import (
    DepthCameraSpec,
    DEPTH_IDEAL,
    DEPTH_STEREO,
    DEPTH_TOF,
    depth_spec_from_text,
    load_depth_spec,
    depth_spec_pinhole_fovy,
    depth_directions,
    depth_to_points,
    depth_return,
)
from .depth_scan import (
    depth_mount_from_mjcf_camera,
    depth_cast,
    depth_scan_cpu,
    BatchedDepthCamera,
)
from .occupancy import VoxelMap, scan_endpoints_world, depth_endpoints_world
from .sensor_scene import SensorScene, ray_ready_dims
