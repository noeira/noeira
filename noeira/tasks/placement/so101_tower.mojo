"""`so101_tower`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `noeira/tasks/families/so101_tower.family`,
`noeira/tasks/scenes/so101_tower.xml` and forward kinematics on it.
2 free slots, 5 regions (0 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from noeira.tasks.placement.table import PlacementTable


struct So101TowerPlacement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 2
    comptime N_REGIONS: Int = 5
    comptime NQ: Int = 20
    comptime NV: Int = 18
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 13
    comptime NSITE: Int = 4
    comptime GRIPPER_SITE: Int = 2  # robot_grasp_center
    comptime N_BASE_QPOS: Int = 6

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        if i == 0:
            return Scalar[DTYPE](0.39477)
        if i == 1:
            return Scalar[DTYPE](-1.908661)
        if i == 2:
            return Scalar[DTYPE](1.544271)
        if i == 3:
            return Scalar[DTYPE](1.342561)
        if i == 4:
            return Scalar[DTYPE](1.258746)
        return Scalar[DTYPE](0.310346)

    @staticmethod
    def base_qpos_jitter[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        if i == 0:
            return Scalar[DTYPE](0.639826)
        if i == 1:
            return Scalar[DTYPE](0.0)
        if i == 2:
            return Scalar[DTYPE](0.0)
        if i == 3:
            return Scalar[DTYPE](0.0)
        if i == 4:
            return Scalar[DTYPE](0.303035)
        return Scalar[DTYPE](0.459539)

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # bowl
        return 3  # brick

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 6
        return 13

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 6
        return 12

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0125)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.060614)
        return Scalar[DTYPE](0.0176777)

    @staticmethod
    def free_park_x[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](11.0)
        return Scalar[DTYPE](11.5)

    @staticmethod
    def free_park_y[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def free_park_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](50.0)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](-0.0125)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.045)
        return Scalar[DTYPE](0.0125)

    @staticmethod
    def region_site(r: Int) -> Int:
        return 3

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.32)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.002)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # desk_top
        if r == 1:
            return True  # desk_left
        if r == 2:
            return True  # desk_right
        if r == 3:
            return True  # desk_brick
        return True  # desk_bowl

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.14)
        if r == 1:
            return Scalar[DTYPE](-0.14)
        if r == 2:
            return Scalar[DTYPE](-0.14)
        if r == 3:
            return Scalar[DTYPE](-0.22)
        return Scalar[DTYPE](-0.19)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.16)
        if r == 1:
            return Scalar[DTYPE](0.06)
        if r == 2:
            return Scalar[DTYPE](-0.16)
        if r == 3:
            return Scalar[DTYPE](-0.24)
        return Scalar[DTYPE](-0.23)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06)
        if r == 1:
            return Scalar[DTYPE](0.06)
        if r == 2:
            return Scalar[DTYPE](0.06)
        if r == 3:
            return Scalar[DTYPE](0.05)
        return Scalar[DTYPE](0.06)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.16)
        if r == 1:
            return Scalar[DTYPE](0.16)
        if r == 2:
            return Scalar[DTYPE](-0.06)
        if r == 3:
            return Scalar[DTYPE](0.21)
        return Scalar[DTYPE](0.19)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def joint_name(k: Int) -> String:
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0
