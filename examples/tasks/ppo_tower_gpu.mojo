"""PPO on `so101_tower` — so101-nexus's recipe on the rig's family.

    pixi run -e nvidia mojo build -I . examples/tasks/ppo_tower_gpu.mojo -o ppo_tower
    ./ppo_tower so101_tower_lift_brick --steps 30000000
    ./ppo_tower so101_tower_cube_in_bowl --steps 100000000 --success-bonus 50
    # -D TASK_PPO_LANES_4096 at build for 4096 lanes, _256 for a smoke run

Flags: --steps, --seed, --lr, --ent-coef, --ent-coef-final, --anneal-steps,
--log-std-init, --reward potential|legacy, --success-bonus, --action absolute|delta,
--init RUN_DIR (with --steps 0: evaluate only), --eval-rounds N (greedy, held-out),
--bc-demos a.demo,b.demo [--bc-updates 2000 --bc-lr 1e-3 --bc-log-std -1] (clone the teacher first),
--checkpoint-every. The driver is `noeira/tasks/ppo_family_driver.run_ppo`;
read its header, and `noeira-docs/SO101_PIXEL_RL_PLAN.md` for why PPO.
"""

from std.sys import argv

from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.ppo_family_driver import run_ppo


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    run_ppo[So101TowerModel, So101TowerConfig](
        args,
        family_path=String("noeira/tasks/families/so101_tower.family"),
        family=String("so101_tower"),
        project=String("so101-tower"),
        driver=String("examples/tasks/ppo_tower_gpu.mojo"),
        default_task=String("so101_tower_lift_brick"),
        shape_w_goal=So101TowerConfig.SHAPE_W_GOAL,
        shape_w_reach=So101TowerConfig.SHAPE_W_REACH,
        goal_margin=So101TowerConfig.GOAL_MARGIN,
        reach_margin=So101TowerConfig.REACH_MARGIN,
    )
