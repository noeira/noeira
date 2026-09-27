"""A pixel student for `so101_tower`, taught by a state PPO teacher (DAgger).

    pixi run -e nvidia mojo build -I . examples/tasks/dagger_tower_pixels.mojo -o dagger_px
    ./dagger_px so101_tower_lift_real_layout --teacher runs/<ppo run> --steps 20000000
    # -D DAGGER_WRIST_ONLY for the wrist camera alone; -D TASK_PPO_LANES_256 for a smoke run

Flags: --teacher RUN_DIR (required), --steps, --beta-steps, --updates, --lr,
--replay ROWS, --seed, --eval-rounds, --init-student CKPT, --png DIR,
--blank-images 1 (the joints-only control). The driver is
`noeira/tasks/pixel_dagger_tower.run_pixel_dagger`; read its header.
"""

from std.sys import argv

from noeira.tasks.pixel_dagger_tower import run_pixel_dagger


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    run_pixel_dagger(args, driver=String("examples/tasks/dagger_tower_pixels.mojo"))
