"""Perception in simulation: sensors as data, mounted on any body of any model.

The point of this package is to choose hardware with the simulator before
buying it, so nothing here is written for one sensor or one robot:

  lidar.mojo        `LidarSpec` (a scan pattern + range window + noise, read
                    from a text file — see `zoo/`), `LidarMount` (a pose on
                    any body), the sensor-frame directions, the noise model
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
