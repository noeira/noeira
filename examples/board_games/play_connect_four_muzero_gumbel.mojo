"""Play Connect Four against a trained Gumbel MuZero checkpoint (SDL UI).

Loads the spatial h/g/f trio saved by `connect_four_muzero_gumbel_spatial.mojo`
(`checkpoint_every` rolling save, or the end-of-run save) and lets you play with
the mouse/keyboard in an SDL window. The agent picks its move with a full
Gumbel-MCTS search over the learned model (deterministic, `gumbel_scale=0.0`) —
the deployed-agent strength, not the bare policy head.

The board is drawn by the env's own `render_board` (the `RenderableEnv`-style
board visual lives in `ConnectFourEnv`, shared with the other play scripts); this
script only overlays the interactive chrome (column selector + status text) and
runs the MuZero search.

You are Red (first), MuZero is Yellow. Controls:
  Mouse hover/click: select + drop in a column
  Left/Right arrows: move the column selector; Space/Return: drop
  R: reset after the game ends; close the window to quit.

The network dims here MUST match the trained checkpoint (CH / NB / BINS / 6×7). If
you retrain with different `CH`/`NB`/`BINS`, update them here too or the load will
fail on a name/shape mismatch. These default to the spatial example's current
config (CH=64, NB=3); for an older CH=32 / 2-block checkpoint set CH=32, NB=2.

Usage (after a training run has written the checkpoint):
    pixi run -e nvidia mojo run -I . examples/board_games/play_connect_four_muzero_gumbel.mojo
    pixi run -e apple  mojo run -I . examples/board_games/play_connect_four_muzero_gumbel.mojo
    pixi run -e apple  mojo run -I . examples/board_games/play_connect_four_muzero_gumbel.mojo --ckpt <run_id>
"""

from std.memory import alloc
from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.checkpoint import load_params_multi
from noeira.deep_agents.muzero.nets_spatial import (
    MZRepNetC4Spatial, MZDynNetC4Spatial, MZPredNetC4Spatial,
)
from noeira.deep_agents.zero.mcts_adapters_mz import (
    MZRepGPU, MZDynGPU, MZPredGPU,
)
from noeira.planners.tree_search import GumbelGPUMCTS, SelfPlay
from noeira.envs.board_games.connect_four.connect_four import ConnectFourEnv
from noeira.render import Renderer2D, SDL_Color
from noeira.render.sdl.sdl_keyboard import get_keyboard_state
from noeira.render.sdl.sdl_scancode import Scancode
from noeira.render.sdl.sdl_mouse import get_mouse_state


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    # ── architecture — MUST match the trained checkpoint ──
    comptime OBS = 126
    comptime ACT = 7
    comptime CH = 64          # must match the trained checkpoint (was 32)
    comptime NB = 3           # residual blocks per net (was 2)
    comptime HH = 6
    comptime WW = 7
    comptime LATENT = CH * HH * WW
    comptime BINS = 51
    # search budget for play (independent of training; more = stronger).
    comptime NUM_SIMS = 200
    comptime MAX_NODES = 256
    comptime MAX_K = 4

    comptime Rep = MZRepNetC4Spatial[CH, HH, WW, NB]
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW, NB]
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW, NB]

    var ckpt = resolve_checkpoint(
        _flag(String("--ckpt"), String("connect_four_muzero_gumbel_spatial.ckpt")),
        String("best"),
    )

    var ctx = DeviceContext()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)

    # ── load the trio ── single-file checkpoint: rep/dyn/pred sections live in
    # ONE file written by the training run's end-of-run + rolling
    # `checkpoint_every` saves (`save_params_multi`); load them back in the same
    # rep→dyn→pred order.
    load_params_multi["gpu", Rep, Dyn, Pred](ckpt, Optional(ctx), rep, dyn, pred)
    rep.set_attr["training"](Scalar[DT](0.0))
    dyn.set_attr["training"](Scalar[DT](0.0))
    pred.set_attr["training"](Scalar[DT](0.0))
    print("loaded checkpoint:", ckpt)
    print("=== Connect Four vs Gumbel MuZero ===")
    print("You are Red (first). MuZero is Yellow.")
    print("Click a column or use arrows + space; R to reset; close to quit.")

    # ── Gumbel planner over the learned model (deterministic play) ──
    var planner = GumbelGPUMCTS[
        1, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](ctx, gamma=1.0, v_min=-1.0, v_max=1.0, gumbel_scale=0.0)
    var rep_a = MZRepGPU[OBS, LATENT, Rep].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, Dyn].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, Pred].make(pred)

    var d_obs = ctx.enqueue_create_buffer[DT](OBS)
    var h_obs = ctx.enqueue_create_host_buffer[DT](OBS)
    var h_pol = ctx.enqueue_create_host_buffer[DT](ACT)
    var h_legal = ctx.enqueue_create_host_buffer[DT](ACT)
    ctx.synchronize()

    var env = ConnectFourEnv[DType.float64]()
    _ = env.reset()
    var mseed = UInt32(12345)

    # ── SDL window + interactive chrome (board itself drawn by the env) ──
    var renderer = Renderer2D(
        width=560, height=530, fps=30, title="Connect Four vs Gumbel MuZero"
    )
    var bg_color = SDL_Color(r=0x11, g=0x11, b=0x44, a=0xFF)
    var red_color = SDL_Color(r=0xFF, g=0x22, b=0x22, a=0xFF)
    var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var win_text_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
    var selector_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
    var status_bg = SDL_Color(r=0x11, g=0x11, b=0x22, a=0xFF)
    var ai_text = SDL_Color(r=0x88, g=0xFF, b=0x88, a=0xFF)

    comptime HUMAN = 0          # Red, moves first
    var cell_size = 80
    var board_cols = 7
    var board_height = 6 * cell_size

    var selected_col = 3
    var prev_left = False
    var prev_right = False
    var prev_space = False
    var prev_return = False
    var prev_r = False
    var prev_mouse_left = False

    var mouse_x_ptr = alloc[Float32](1)
    var mouse_y_ptr = alloc[Float32](1)
    mouse_x_ptr[] = Float32(0)
    mouse_y_ptr[] = Float32(0)
    var numkeys_ptr = alloc[Int32](1)
    numkeys_ptr[] = 0

    while renderer.begin_frame_with_color(bg_color):
        var keys = get_keyboard_state(numkeys_ptr)
        var cur_left = Bool(keys[Int(Scancode.SCANCODE_LEFT)])
        var cur_right = Bool(keys[Int(Scancode.SCANCODE_RIGHT)])
        var cur_space = Bool(keys[Int(Scancode.SCANCODE_SPACE)])
        var cur_return = Bool(keys[Int(Scancode.SCANCODE_RETURN)])
        var cur_r = Bool(keys[Int(Scancode.SCANCODE_R)])

        var mouse_buttons = get_mouse_state(
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_x_ptr),
            rebind[Pointer[Float32, MutAnyOrigin]](mouse_y_ptr),
        )
        var cur_mouse_left = (Int(mouse_buttons.value) & 1) != 0
        var mouse_x = Int(mouse_x_ptr[])
        var mouse_col = mouse_x // cell_size
        var mouse_on_board = mouse_col >= 0 and mouse_col < board_cols
        if mouse_on_board:
            selected_col = mouse_col

        var game_over = env.done

        # Reset after a finished game.
        if cur_r and not prev_r and game_over:
            _ = env.reset()
            selected_col = 3

        # ── Human turn (Red / player 0) ──
        if not game_over and env.current_player() == HUMAN:
            if cur_left and not prev_left and selected_col > 0:
                selected_col -= 1
            if cur_right and not prev_right and selected_col < board_cols - 1:
                selected_col += 1

            var action = -1
            if (cur_space and not prev_space) or (
                cur_return and not prev_return
            ):
                action = selected_col
            if cur_mouse_left and not prev_mouse_left and mouse_on_board:
                action = mouse_col

            if action >= 0:
                var legal = env.legal_action_mask()
                if action < legal.byte_length() and legal[action]:
                    _ = env._step_impl(action)

        # ── MuZero turn (Yellow / player 1): Gumbel search, argmax-legal ──
        if not game_over and env.current_player() != HUMAN:
            var legal = env.legal_action_mask()
            var obs = env.get_obs_list()
            for j in range(OBS):
                h_obs.unsafe_ptr()[j] = Scalar[DT](Float64(obs[j]))
            for a in range(ACT):
                h_legal.unsafe_ptr()[a] = (
                    Scalar[DT](1.0) if legal[a] else Scalar[DT](0.0)
                )
            ctx.enqueue_copy(d_obs, h_obs)
            ctx.enqueue_copy(planner.legal_mask_view(), h_legal)
            var obs_t = LayoutTensor[
                DT, Layout.row_major(1, OBS), MutAnyOrigin
            ](d_obs)
            planner.search_gpu[
                type_of(rep_a), type_of(dyn_a), type_of(pred_a)
            ](
                ctx, rep_a, dyn_a, pred_a, obs_t,
                apply_legal=True, k_actual=MAX_K, rng_seed=mseed,
            )
            mseed += UInt32(1)
            ctx.enqueue_copy(h_pol, planner.policies_view())
            ctx.synchronize()
            var best = -1
            var bv = -1.0e30
            for a in range(ACT):
                if legal[a] and Float64(h_pol.unsafe_ptr()[a]) > bv:
                    bv = Float64(h_pol.unsafe_ptr()[a])
                    best = a
            if best < 0:
                best = 0
            print("MuZero plays column", best)
            _ = env._step_impl(best)

        # ── Render: env draws the board, we overlay selector + status ──
        var selector_cx = selected_col * cell_size + cell_size // 2
        if not env.done and env.current_player() == HUMAN:
            renderer.draw_circle(selector_cx, 22, 18, red_color, filled=True)
            renderer.draw_line(
                selector_cx, 42, selector_cx - 8, 35, selector_color, 2
            )
            renderer.draw_line(
                selector_cx, 42, selector_cx + 8, 35, selector_color, 2
            )

        env.render_board(renderer)

        # Status bar.
        renderer.draw_rect(0, 50 + board_height, board_cols * cell_size, 50, status_bg)
        var gr = env.game_result()
        if gr == 0:
            if env.current_player() == HUMAN:
                renderer.draw_text(
                    "Your turn (Red)", 200, 50 + board_height + 20, text_color
                )
            else:
                renderer.draw_text(
                    "MuZero thinking...", 195, 50 + board_height + 20, ai_text
                )
        elif gr == HUMAN + 1:
            renderer.draw_text(
                "You win!  (R to reset)",
                175, 50 + board_height + 20, win_text_color,
            )
        elif gr == 3:
            renderer.draw_text(
                "Draw!  (R to reset)",
                195, 50 + board_height + 20, win_text_color,
            )
        else:
            renderer.draw_text(
                "MuZero wins!  (R to reset)",
                160, 50 + board_height + 20, win_text_color,
            )

        renderer.flip()

        prev_left = cur_left
        prev_right = cur_right
        prev_space = cur_space
        prev_return = cur_return
        prev_r = cur_r
        prev_mouse_left = cur_mouse_left

    mouse_x_ptr.free()
    mouse_y_ptr.free()
    numkeys_ptr.free()
    renderer.close()
    print("=== Done ===")
