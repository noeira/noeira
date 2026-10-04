# Ground navigation — robot-agnostic: footprints -> costmap -> MPPI.
#
#   costmap   Footprint, NavGrid (exact clearance + Dijkstra navigation
#             function: free-path distance to the goal, no local minima)
#   scene     footprints_from_model: the colliding geoms of a parsed MJCF
#             scene, minus the robot, projected onto the ground
#   unicycle  ResponseMap (commanded -> achieved velocities, measured),
#             UnicycleNavCallback (RolloutCallbackCPU), UnicycleNavigator
#             (MPPICPU over it, receding horizon)
#
# First user: the G1 room (projects/g1, BFM_ZERO_ROOM_PLAN R5b), whose robot
# adapter maps (v, w) to a BFM-Zero prompt. Nothing here knows about it.

from .costmap import Footprint, NavGrid, NAV_UNREACHABLE, FP_RECT, FP_CIRCLE
from .scene import footprints_from_model, footprint_of_geom, SceneFootprints
from .unicycle import (
    ResponseMap, NavParams, nav_params_default, UnicycleNavCallback,
    UnicycleNavigator,
)
