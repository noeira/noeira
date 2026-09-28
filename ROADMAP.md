# noeira Roadmap

noeira started as a reinforcement-learning framework and grew into an
end-to-end Physical AI stack: simulation, learning, perception and deployment,
from simulator to robot. The first section is what that growth added; the
rest is the RL and infrastructure history it was built on.

## Completed — Physical AI (2026-08 → 2026-09)

### Robots and data
- [x] SO-101 over its Feetech servo bus (`robot/`): diagnostics, calibration (EEPROM writes opt-in), torque control, opt-in arming
- [x] Cameras (`vision/`): threaded capture, ChArUco intrinsics, camera-to-base extrinsics, the OpenCV shim
- [x] Teleoperation recorded to LeRobot v3 datasets (Parquet + H.264), checkpointed and resumable; pushed to the Hugging Face Hub; imported back natively
- [x] Jetson Orin NX deployment (`-e jetson`, sm_87): ACT's closed loop at 30.0 Hz on the arm
- [x] Projects and runs (`RunContext`, policy promotion, push / pull) and noeira cloud at cloud.noeira.ai

### Imitation, VLAs and learning from demonstrations
- [x] ACT (LeRobot parity; trained on SO-101 datasets and LIBERO; deployed on the arm)
- [x] SmolVLA — inference and fine-tuning in Mojo; running on the arm at 27 Hz; first grasp-and-place on the arm on 2026-09-20 (cube picked and dropped into the bowl), roughly one attempt in five by informal count [experimental — needs more and better data]
- [x] Behaviour cloning (`deep_agents/bc/`)
- [x] HIL-SERL (RLPD demo prefix + BC term) and DAgger-style interventions (`deep_agents/demos/`, `deep_agents/hil_serl/`) [experimental — first lift in simulation]
- [x] Forward-Backward zero-shot RL; BFM-Zero reproduced on the Unitree G1

### Simulation and tasks
- [x] MuJoCo 3.12 parity audit: no unserved feature across 83 Menagerie models and DeepMind Control
- [x] DeepMind Control suite, every task parity-gated against MuJoCo [training not yet shown]
- [x] The task layer (`tasks/`): families, goals, placement, batched multi-task evaluation
- [x] LIBERO (`envs/libero/`): 23 families, 129 tasks, demos imported and re-rendered, batched OSC_POSE evaluation
- [x] Ray-traced batched cameras, ImGui viewers, the physics studio

### Toolchain
- [x] Mojo 1.1.0 / MAX 26.6.0, pinned; `linux-aarch64` for the Jetson
- [x] MAX flash attention routed for the fused cross-attention on NVIDIA

## Completed — RL and infrastructure

### Environments - Native Mojo (Tabular)

- [x] GridWorld - 2D navigation (5x5 default)
- [x] FrozenLake - Slippery grid with holes (4x4)
- [x] CliffWalking - Cliff avoidance task (4x12)
- [x] Taxi - Pickup/dropoff with 500 states

### Environments - Native Mojo (Classic Control)

- [x] CartPole (Native) - Pure Mojo, 145x faster than Gymnasium, GPU batch support
- [x] MountainCar (Native) - Physics matching Gymnasium MountainCar-v0, tile coding support
- [x] Acrobot (Native) - Two-link pendulum with RK4 integration, tile coding + polynomial features
- [x] Pendulum (Native) - Continuous actions, V1 (CPU) and V2 (GPU batch) implementations

### Environments - Native Mojo (2D Physics Engine)

- [x] LunarLander - Custom physics2d engine, discrete + continuous actions, GPU batch, terrain + flame particles
- [x] BipedalWalker - Custom physics2d with revolute joints, 24D obs, 4D continuous actions, lidar, normal/hardcore modes, GPU batch
- [x] CarRacing - Custom physics2d with slip-based tire friction, procedural tracks, 3D continuous actions, GPU batch

### Environments - Native Mojo (MuJoCo-Style, physics3d Engine)

- [x] HalfCheetah - 9 bodies, 8 joints, 6 actuators, 17D obs, GPU batch
- [x] Hopper - 5 bodies, 4 joints, 3 actuators, 11D obs, GPU batch
- [x] Ant - 13 bodies, 9 joints, 8 actuators, 27D obs, GPU batch
- [x] Walker2d - 7 bodies, 6 joints, 6 actuators, 17D obs, GPU batch
- [x] Swimmer - 5 bodies, 2 joints, 2 actuators, 8D obs, GPU batch
- [x] Humanoid - 17 bodies, 16 joints, 17 actuators, 376D obs, GPU batch
- [x] HumanoidStandup - Humanoid with standing reward, GPU batch
- [x] InvertedPendulum - 2 bodies, 1 joint, 1 actuator, 4D obs, GPU batch
- [x] InvertedDoublePendulum - 3 bodies, 2 joints, 2 actuators, 11D obs, GPU batch

### Environments - Gymnasium Wrappers (`envs/gymnasium/`)

- [x] Generic Gymnasium wrapper - Works with any Gymnasium environment
- [x] **Classic Control**: CartPole, MountainCar, Pendulum, Acrobot
- [x] **Box2D**: LunarLander, BipedalWalker, CarRacing
- [x] **Toy Text**: FrozenLake, Taxi, Blackjack, CliffWalking
- [x] **MuJoCo**: HalfCheetah, Ant, Humanoid, Walker2d, Hopper, Swimmer, InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher

### Algorithms - TD Methods

- [x] Q-Learning - Off-policy TD learning
- [x] SARSA - On-policy TD learning
- [x] Expected SARSA - Lower variance than SARSA
- [x] Double Q-Learning - Reduces overestimation bias

### Algorithms - Multi-step Methods

- [x] N-step SARSA - Configurable n-step returns
- [x] SARSA(lambda) - Eligibility traces with replacing traces
- [x] Monte Carlo - First-visit MC

### Algorithms - Model-based

- [x] Dyna-Q - Q-Learning with model-based planning
- [x] Priority Sweeping - Prioritized updates by TD error

### Environments - Arcade Games (`envs/arcade_games/`) [In Progress]

- [x] **Pong** - Native GPU engine, clean obs (6D) + pixel obs (4x84x84), 3 actions
- [x] **Breakout** - Native GPU engine, 6x14 brick grid, clean obs (7D) + pixel obs, 4 actions
- [x] **Space Invaders** - Native GPU engine, 5x11 alien grid, clean obs (10D) + pixel obs, 4 actions
- [ ] Freeway, Enduro, Qbert, Asteroids (planned — same template)

### Environments - Atari 2600 Emulator (`envs/atari/`)

- [x] **6502 CPU emulator** - Full fetch-decode-execute, all official opcodes
- [x] **TIA graphics chip** - Playfield, sprites, ball, collision detection
- [x] **RIOT** - Timer, I/O ports
- [x] **ROM support** - Pong, Breakout, Space Invaders game definitions
- [x] **Frame rendering** - BGRA, RGB, grayscale output modes
- [x] **SDL3 renderer** - Interactive play with keyboard input

### Infrastructure

- [x] Replay Buffer - Circular buffer for experience replay
- [x] Prioritized Replay Buffer - Samples by TD error priority (sum-tree)
- [x] Q-Learning with Replay - Off-policy learning with replay buffer
- [x] Q-Learning with PER - Off-policy learning with prioritized replay
- [x] Continuous Replay Buffer - For continuous state/action algorithms
- [x] GPU Replay Buffer - Device-side circular buffer for GPU training
- [x] GPU Prioritized Replay Buffer - CPU sum-tree + GPU data storage
- [x] N-Step Buffer - N-step returns accumulation (for Rainbow)
- [x] Sequence Replay Buffer - Contiguous sequence sampling for model-based RL (TD-MPC2)
- [x] GPU Sequence Replay Buffer - Device-side sequence buffer

### Function Approximation

- [x] Tile Coding - Multi-dimensional overlapping tilings for continuous states
- [x] Linear Function Approximation - Polynomial and RBF feature extractors
- [x] Tiled Q-Learning, SARSA, SARSA(lambda) agents
- [x] Linear Q-Learning, SARSA, SARSA(lambda) agents

### Policy Gradient Methods

- [x] REINFORCE - Monte Carlo policy gradient with optional baseline + entropy variant
- [x] Actor-Critic - One-step TD, Actor-Critic(lambda) with traces, A2C with n-step returns
- [x] GAE (Generalized Advantage Estimation) - Exponentially-weighted TD residuals
- [x] PPO - Clipped surrogate objective, minibatch variant

### Continuous Control (Linear FA)

- [x] DDPG - Deterministic actor + Q-critic with target networks
- [x] TD3 - Twin critics, delayed policy, target smoothing
- [x] SAC - Stochastic Gaussian policy, max entropy, auto alpha tuning

### Deep Learning Framework (`nn/`)

The storage framework that replaced the legacy `nn` in the sunset: modules own
their weights, and gradients come from each module's own VJP.

- [x] **Module trait** — `forward` and `vjp` over tensor packs, compile-time dimensions, CPU and GPU from one source
- [x] **Param** — each weight owns its value and gradient tensors; parameters are discovered by reflection (`for_each_param`), so the optimizer and checkpoints need no registration
- [x] **Gradients by composition** — every primitive writes its VJP; combinators chain them at compile time. Reverse mode with no tape and no runtime graph
- [x] **Primitives (70+)** — Linear, Conv2D / Conv2DTranspose, pooling, LayerNorm / RMSNorm / BatchNorm / SimNorm, attention (self, cross, masked; fused, and MAX's flash attention on NVIDIA), RoPE, SwiGLU, embeddings, LSTM / GRU cells, NoisyLinear, BlockLinear (ensembles in one kernel), RL ops (rsample, symlog, dueling heads)
- [x] **Fused primitives** — `LinearAct` and its `LinearReLU` / `LinearTanh` / … aliases (matmul + bias + activation in one epilogue kernel), `LayerNormAct`. Fusion is chosen by the primitive you use, not by a compiler pass; convolutions have no fused-activation form
- [x] **Combinators** — Sequential, Residual, ProjectedResidual, Parallel, Repeat, SkipConcat, Tokenwise, StopGrad, and `ComputeGraph`, a named-node DAG for loss graphs that are not a chain
- [x] **Models** — ResNet, ResNet-18 (torchvision-compatible weights), GPT, ViT, transformers
- [x] **Optimizers** — SGD, Adam / AdamW with a parameter arena and grouped multi-tensor apply, gradient clipping, LR schedules
- [x] **Losses** — MSE, cross-entropy, soft cross-entropy, sequence cross-entropy, two-hot, BCE with logits, Gaussian NLL, gradient penalty
- [x] **Training** — supervised and autoregressive `Trainer`s, AMP (bf16 activations, fp32 master weights), CUDA-graph capture
- [x] **Checkpoints v2** — every `Param` is `Saveable`; safetensors import for published weights
- [x] **Datasets** — MNIST, CIFAR-10, TinyShakespeare, LeWM Push-T

### Deep RL Agents (`deep_agents/`)

One package per algorithm, each an `XxxAgent` facade over a trainer and a shared
training driver (the legacy config-driven `GenericXxxAgent` design was removed).

- [x] **Value-based** — DQN (Double, Dueling, Noisy, PER variants), C51, Rainbow
- [x] **Off-policy actor-critic** — DDPG, TD3, SAC, REDQ, REDQ-OFE
- [x] **On-policy** — PPO (continuous and discrete, GPU-batched), A2C
- [x] **Model-based** — MBPO, TD-MPC2, DreamerV3, Dreamer 4
- [x] **Search** — AlphaZero, MuZero, EfficientZero V2 (shared `zero/` package, GPU MCTS)
- [x] **Imitation and VLAs** — behaviour cloning, ACT, SmolVLA
- [x] **From demonstrations** — HIL-SERL (RLPD pinned demo prefix + BC term), DAgger recorders, the `.demo` format
- [x] **Zero-shot** — Forward-Backward and FB-CPR (BFM-Zero on the Unitree G1)

### Deep RL Shared Infrastructure (`deep_agents/training/`, `data/`)

- [x] **Training drivers** — off-policy and on-policy, continuous and discrete, CPU and GPU (`run_offpolicy_*` / `run_onpolicy_*`), over a `BatchedEnv` trait
- [x] **Trainer blocks** — sampling, acting and update blocks the drivers compose; CUDA-graph-capturable steps
- [x] **Replay** — `StoreReplay` / `StoreReplayGpu` on the trajectory store: uniform and prioritized, ERE, uint8 observations, device-side sampling, a pinned demonstration prefix
- [x] **Runs** — `RunContext` records each run's checkpoints and metrics under a project, mirrored to noeira cloud

### 3D Physics Engine (`physics3d/`)

- [x] **Generalized coordinates dynamics** - MuJoCo-inspired joint-space representation
- [x] **Mass matrix** - CRBA (Composite Rigid Body Algorithm), sparse variants, LDL/LU decomposition
- [x] **Bias forces** - RNE (Recursive Newton-Euler) for Coriolis + gravity
- [x] **Jacobians** - Contact Jacobians, analytical Jacobians, composite inertia
- [x] **Joint types** - FREE (7 DOF), BALL (4 DOF), SLIDE (1 DOF), HINGE (1 DOF)
- [x] **Integrators** - Euler, ImplicitFast (default), Implicit (full RNE velocity derivative), RK4
- [x] **Constraint solvers** - PGS, Newton, CG, Island-based PGS with early termination
- [x] **Collision detection** - Sphere/capsule/box narrow-phase, Sweep-and-Prune broadphase, CPU + GPU
- [x] **Constraint building** - Contact, equality, tendon constraints, CPU + GPU
- [x] **Compile-time model specs** - BodySpec, JointSpec, GeomSpec, ActuatorSpec, EqualitySpec, TendonSpec traits
- [x] **ModelDef compositor** - Variadic iteration for N-body composition
- [x] **MJCF XML parser** - XML -> FlatModelDef -> Model/Data pipeline
- [x] **GPU support** - Forward kinematics, body velocities, collision, constraint building all have GPU paths
- [x] **Validation** - 75 test files comparing against MuJoCo reference

### 2D Physics Engine (`physics2d/`)

- [x] **Batched GPU physics** - Strided [BATCH, STATE_SIZE] layout for parallel simulation
- [x] **Impulse solver** - Velocity + position level contact resolution with warm-starting
- [x] **Revolute joints** - Motor control, spring damping, angle limits
- [x] **Terrain collision** - Flat ground and edge terrain detection
- [x] **Articulated chains** - Multi-body support for Hopper, Walker, Cheetah configurations
- [x] **Car physics** - Slip-based tire friction, track tile lookup, fused GPU kernel
- [x] **Lidar sensors** - Ray-cast distance sensing
- [x] **PhysicsKernel** - One-call step_gpu() for full physics step

### 3D Math Library (`math3d/`)

- [x] Vec3, Quat, Mat3, Mat4 with GPU variants

### Rendering (`render/`)

- [x] **SDL3 FFI bindings** - Complete bindings (38 files: video, render, GPU, events, keyboard, mouse, audio, etc.)
- [x] **Renderer2D** - SDL3 2D CPU rasterizer (rect, line, circle, polygon, text)
- [x] **Renderer3D** - GPU-accelerated 3D renderer using SDL3 GPU API
  - Blinn-Phong lighting with up to 4 lights
  - Shadow mapping
  - Procedural checkerboard ground with reflections
  - Gradient skybox
  - GPU bitmap font atlas for HUD text
  - Mesh caching (sphere, box, capsule LRU cache)
  - Deferred draw commands
  - MSL Metal shaders (solid, ground, line, shadow, reflection, skybox, text)
- [x] **Interactive camera** - Orbit, pan, zoom (mouse), camera switching (1-9 keys)
- [x] **Playback control** - Pause (Space), step (->), reset camera (R)
- [x] **Video recording** - MP4/GIF via Python imageio (V key toggle)
- [x] **Screenshot** - GPU readback (S key)
- [x] **Colors and shapes** - 30+ named colors, 10+ 2D shape factories, 3D wireframe generators
- [x] **Camera types** - 2D Camera + RotatingCamera, 3D Camera3D with perspective projection

### Vectorized Environments

- [x] VecCartPoleEnv - SIMD-based parallel CartPole (SoA layout, auto-reset)

### Infrastructure

- [x] Logging/Metrics - EpisodeMetrics, TrainingMetrics with convergence/success tracking
- [x] Hyperparameter Search - Grid and random search with CSV export
- [x] Learning rate scheduling - Linear annealing in Deep PPO

## In Progress / Next Steps

### Physical AI
- [ ] A measured success rate for ACT on the real arm
- [ ] SmolVLA on the arm: a measured success rate, and more / better demonstrations to raise it (≈20% by informal count, 2026-09-20)
- [ ] HIL-SERL and DAgger on the real arm, from the simulation recipe
- [ ] LIBERO success rates beyond the current baselines (ACT 31/200 vs a vision-free kNN 144/200 on libero_goal)
- [ ] Training results on DeepMind Control

### GPU Optimization

- [ ] MMA tensor core matmul for all platforms (currently Apple Silicon optimized)
- [x] Mixed precision (AMP) in the supervised `Trainer`
- [ ] Mixed precision across the RL trainers
- [ ] Multi-GPU support

### Infrastructure Improvements

- [ ] Cosine annealing and warmup LR schedulers
- [ ] Curriculum learning framework (CurriculumScheduler trait exists)
- [ ] Population-based training
- [ ] TensorBoard-style metric visualization
- [ ] Parallel training across multiple environments

### Environments

- [ ] More arcade games (Freeway, Enduro, Qbert, Asteroids)
- [ ] Custom environment builder
- [ ] POMDP benchmark environments

## Future Exploration

> Ideas for future development, roughly prioritized by potential impact.

### Offline / Batch RL

Learning from fixed datasets without environment interaction.

- [ ] **CQL (Conservative Q-Learning)** - Penalizes Q-values for out-of-distribution actions
- [ ] **IQL (Implicit Q-Learning)** - Avoids explicit policy evaluation
- [ ] **Decision Transformer** - Treats RL as sequence modeling (returns-conditioned)
- [ ] **BCQ (Batch-Constrained Q-Learning)** - Constrains policy to data support
- [ ] **AWR (Advantage Weighted Regression)** - Simple offline-compatible algorithm
- [ ] Dataset infrastructure - D4RL format loading/saving

### Distributional RL

Model full distribution of returns instead of just expected value.

- [x] **C51** - Categorical distribution over returns (51 atoms)
- [x] **Rainbow** - Combines 6 DQN improvements (C51, Double, PER, Dueling, NoisyNets, N-step)
- [ ] **QR-DQN** - Quantile regression for distributional RL
- [ ] **IQN (Implicit Quantile Networks)** - Sample quantile fractions

### Exploration Enhancements

For sparse reward and hard exploration problems.

- [x] **NoisyNets** - Learnable parametric noise in network weights (NoisyLinear layer)
- [ ] **ICM (Intrinsic Curiosity Module)** - Prediction error as intrinsic reward
- [ ] **RND (Random Network Distillation)** - Simpler curiosity-driven exploration
- [ ] **Bootstrapped DQN** - Ensemble for uncertainty estimation

### Recurrent Policies (POMDPs)

For partial observability and memory-dependent tasks.

- [x] **LSTM / GRU cells** in `nn` (LSTMCell, GRUCell, LSTM sequence)
- [ ] **R2D2** - Recurrent DQN with burn-in and stored hidden states

### Multi-Agent RL

Cooperative and competitive multi-agent settings.

- [ ] **MADDPG** - Multi-agent DDPG with centralized critic
- [ ] **QMIX** - Value decomposition for cooperative agents
- [ ] **IPPO** - Independent PPO baseline
- [ ] Simple multi-agent environments

### Model-Based Deep RL (Beyond TD-MPC2)

- [x] **DreamerV3** - RSSM world model, actor-critic in imagination, categorical latent states [experimental — performance tuning in progress]
- [x] **MBPO (Model-Based Policy Optimization)** - Short rollouts from learned model
- [ ] **World Models** - VAE + MDN-RNN for latent dynamics

### Architecture Extensions

- [x] Transformer blocks for sequence modeling (GPT, ViT, cross-attention)
- [x] Imitation learning (behaviour cloning, DAgger-style interventions, ACT, SmolVLA)

### Quick Wins

- [ ] **HER (Hindsight Experience Replay)** - Works with existing replay infrastructure
- [ ] **Soft Q-Learning** - Max entropy with discrete actions
- [x] **n-step DQN** - Multi-step returns for DQN (implemented in Rainbow via NStepBuffer)

## Algorithm Summary

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| Q-Learning | TD | Off-policy, max Q(s',a') |
| SARSA | TD | On-policy, Q(s',a') |
| Expected SARSA | TD | E[Q(s',a')], lower variance |
| Double Q-Learning | TD | Two Q-tables, reduces overestimation |
| N-step SARSA | Multi-step | n-step returns |
| SARSA(lambda) | Eligibility | Trace decay |
| Monte Carlo | Episode | Complete episode returns |
| Dyna-Q | Model-based | Planning with learned model |
| Priority Sweeping | Model-based | Prioritized planning |
| Q-Learning + Replay | Replay | Experience replay buffer |
| Q-Learning + PER | Replay | Prioritized experience replay |
| Tiled Q-Learning/SARSA | Function Approx | Tile coding for continuous states |
| Linear Q-Learning/SARSA | Function Approx | Polynomial, RBF features |
| REINFORCE | Policy Gradient | Monte Carlo policy gradient |
| Actor-Critic / A2C | Policy Gradient | TD-based + n-step returns |
| PPO | Policy Gradient | Clipped surrogate, stable updates |
| DDPG (Linear) | Continuous | Deterministic policy + Q-critic |
| TD3 (Linear) | Continuous | Twin critics + delayed updates |
| SAC (Linear) | Continuous | Stochastic + entropy + auto alpha |
| Deep DQN | Deep RL | Neural Q-function + target network |
| Deep Double DQN | Deep RL | Reduced overestimation |
| Deep Dueling DQN | Deep RL | V(s) + A(s,a) streams |
| Deep DQN + PER | Deep RL | Priority sampling by TD error |
| Noisy DQN | Deep RL | NoisyLinear, no epsilon-greedy |
| C51 | Deep RL (Distributional) | Categorical distribution (51 atoms) |
| Rainbow | Deep RL (Distributional) | C51 + Double + PER + Dueling + Noisy + N-step |
| Deep DDPG | Deep RL | Deterministic actor + Q-critic |
| Deep TD3 | Deep RL | Twin critics + delayed + smoothing |
| Deep SAC | Deep RL | Stochastic + entropy + auto alpha |
| Deep A2C | Deep RL | Actor-Critic with GAE |
| Deep PPO | Deep RL | Clipped surrogate + LR anneal + KL stop |
| Deep PPO Continuous | Deep RL | Unbounded Gaussian, GPU training |
| TD-MPC2 | Model-Based RL | World model + MPPI + distributional |
| DreamerV3 | Model-Based RL | RSSM + imagination + categorical latents |
| MuZero | Model-Based RL | Learned model + MCTS + distributional |
| Dreamer 4 | Model-Based RL | Shortcut-forcing world model + imagination training |
| MBPO | Model-Based RL | Short branched rollouts from a model ensemble |
| Behaviour cloning | Imitation | MLP regression on demonstrations, normalised inputs |
| ACT | Imitation | CVAE + action chunking + temporal ensemble |
| SmolVLA | VLA | Vision-language backbone + flow-matching action expert |
| HIL-SERL | From demonstrations | RLPD pinned demo prefix + BC term, interventions |
| Forward-Backward | Zero-shot RL | Successor-measure factorisation, task inferred from reward |
