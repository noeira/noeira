// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkBaseLinks from './src/plugins/remark-base-links.mjs';

// noeira.ai keeps its root for the marketing site; the docs live under /docs.
const BASE = '/docs';

// https://astro.build/config
export default defineConfig({
	// Canonical origin. Required by @astrojs/sitemap (bundled with Starlight),
	// which silently skips without it, and used for canonical <link> tags.
	site: 'https://noeira.ai',
	base: BASE,
	// Built INTO dist/docs so the Worker's asset paths match its URL paths:
	// a request for /docs/start/why/ is looked up as dist/docs/start/why/.
	// dist/ itself carries the root files (robots.txt, _redirects) — see root/.
	outDir: './dist/docs',
	// Content links are written root-relative (/tooling/monitor/); this mounts
	// them under BASE, since Starlight only does that for its own sidebar.
	markdown: { remarkPlugins: [[remarkBaseLinks, { base: BASE }]] },
	integrations: [
		starlight({
			title: 'noeira',
			description:
				'Dream in simulation. Act in the world. An end-to-end Physical AI stack in Mojo: physics, learning, perception and deployment, from simulator to robot.',
			logo: {
				// Full lockup (mark + wordmark), so it replaces the title text.
				// Starlight's `dark`/`light` name the THEME the file is shown in:
				// `dark` needs light-on-transparent, `light` needs dark-on-transparent.
				// Wordmarks are outlined paths — no font dependency.
				dark: './src/assets/noeira-logo-transparent-dark-nobaseline.svg',
				light: './src/assets/noeira-logo-light-nobaseline.svg',
				alt: 'noeira',
				replacesTitle: true,
			},
			favicon: '/favicon.svg',
			head: [{ tag: 'meta', attrs: { name: 'theme-color', content: '#0E1117' } }],
			customCss: ['./src/styles/noeira.css'],
			// Adds og:image + JSON-LD on top of Starlight's own head tags.
			components: { Head: './src/components/Head.astro' },
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/noeira/noeira',
				},
			],
			expressiveCode: {
				themes: ['github-dark-default', 'github-light'],
				styleOverrides: {
					borderRadius: '0.625rem',
					borderColor: 'var(--sl-color-hairline)',
					codeFontFamily: 'var(--sl-font-mono)',
					uiFontFamily: 'var(--sl-font)',
					codeBackground: 'var(--noeira-surface)',
					frames: {
						editorTabBarBackground: 'var(--noeira-surface-raised)',
						editorActiveTabIndicatorBottomColor: 'var(--sl-color-text-accent)',
						terminalTitlebarBackground: 'var(--noeira-surface-raised)',
						terminalBackground: 'var(--noeira-surface)',
					},
				},
			},
			// Grouped by what the project does, simulation to robot. A section gets
			// its entry when its first page lands — no empty placeholder groups.
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'Why noeira', slug: 'start/why' },
						{ label: 'Installation', slug: 'start/installation' },
						{ label: 'Quickstart: tabular', slug: 'start/quickstart-tabular' },
						{ label: 'Quickstart: GPU training', slug: 'start/quickstart-gpu' },
						{
							label: 'Primers',
							collapsed: true,
							items: [{ label: 'RL in five minutes', slug: 'start/rl-in-five-minutes' }],
						},
					],
				},
				{
					label: 'Concepts',
					items: [
						{ label: 'The stack', slug: 'concepts/architecture' },
						{ label: 'Compile-time composition', slug: 'concepts/compile-time-composition' },
						{ label: 'Traits', slug: 'concepts/traits' },
						{ label: 'CPU and GPU paths', slug: 'concepts/cpu-gpu' },
						{ label: 'Checkpoints', slug: 'concepts/checkpoints' },
						{ label: 'Projects and runs', slug: 'concepts/projects' },
					],
				},
				{
					label: 'Simulation',
					items: [
						{
							label: 'Physics',
							items: [
								{ label: '3D engine', slug: 'physics/physics3d' },
								{ label: 'Collision', slug: 'physics/collision' },
								{ label: 'Constraints and tendons', slug: 'physics/constraints' },
								{ label: '2D engine', slug: 'physics/physics2d' },
								{ label: 'Validation', slug: 'physics/validation' },
							],
						},
						{
							label: 'Environments',
							collapsed: true,
							items: [
								{ label: 'Overview', slug: 'environments' },
								{ label: 'Tabular', slug: 'environments/tabular' },
								{ label: 'Classic control', slug: 'environments/classic-control' },
								{ label: '2D physics', slug: 'environments/2d-physics' },
								{ label: '3D locomotion', slug: 'environments/3d-locomotion' },
								{ label: '3D manipulation', slug: 'environments/3d-manipulation' },
								{ label: 'Arcade games', slug: 'environments/arcade' },
								{ label: 'Atari 2600', slug: 'environments/atari' },
								{ label: 'Board games', slug: 'environments/board-games' },
								{ label: 'Procgen & Craftax', slug: 'environments/procgen' },
								{ label: 'DeepMind Control', slug: 'environments/dm-control' },
								{ label: 'LIBERO', slug: 'environments/libero' },
								{ label: 'Robots and humanoids', slug: 'environments/robots' },
								{ label: 'Gymnasium wrappers', slug: 'environments/gymnasium' },
								{ label: 'Writing your own', slug: 'environments/custom' },
							],
						},
						{
							label: 'Tasks',
							collapsed: true,
							items: [
								{ label: 'The task layer', slug: 'tasks' },
								{ label: 'Goals', slug: 'tasks/goals' },
								{ label: 'Placement and resets', slug: 'tasks/placement' },
								{ label: 'Writing a family', slug: 'tasks/writing-a-family' },
							],
						},
					],
				},
				{
					label: 'Learning',
					items: [
						{
							label: 'Algorithms',
							items: [
							{ label: 'Overview', slug: 'algorithms' },
							{
								label: '1 · Tabular and linear',
								collapsed: true,
								items: [
									{ label: 'TD methods', slug: 'algorithms/tabular/td-methods' },
									{ label: 'Function approximation', slug: 'algorithms/tabular/function-approximation' },
									{ label: 'Classical policy gradient', slug: 'algorithms/tabular/policy-gradient' },
								],
							},
							{
								label: '2 · Value-based',
								collapsed: true,
								items: [
									{ label: 'DQN', slug: 'algorithms/dqn' },
									{ label: 'C51 and Rainbow', slug: 'algorithms/rainbow' },
								],
							},
							{
								label: '3 · Policy gradient',
								collapsed: true,
								items: [
									{ label: 'A2C', slug: 'algorithms/a2c' },
									{ label: 'PPO', slug: 'algorithms/ppo' },
								],
							},
							{
								label: '4 · Continuous control',
								collapsed: true,
								items: [
									{ label: 'DDPG', slug: 'algorithms/ddpg' },
									{ label: 'TD3', slug: 'algorithms/td3' },
									{ label: 'SAC', slug: 'algorithms/sac' },
									{ label: 'REDQ', slug: 'algorithms/redq' },
								],
							},
							{
								label: '5 · Model-based',
								collapsed: true,
								items: [
									{ label: 'MBPO', slug: 'algorithms/mbpo' },
									{ label: 'TD-MPC2', slug: 'algorithms/tdmpc2' },
									{ label: 'DreamerV3', slug: 'algorithms/dreamerv3' },
									{ label: 'Dreamer 4', slug: 'algorithms/dreamer4' },
								],
							},
							{
								label: '6 · Planning (zero-series)',
								collapsed: true,
								items: [
									{ label: 'AlphaZero', slug: 'algorithms/alphazero' },
									{ label: 'MuZero', slug: 'algorithms/muzero' },
									{ label: 'EfficientZero V2', slug: 'algorithms/efficient-zero-v2' },
								],
							},
							{
								label: '7 · Imitation and VLAs',
								collapsed: true,
								items: [
									{ label: 'Overview', slug: 'algorithms/imitation' },
									{ label: 'Behaviour cloning', slug: 'algorithms/imitation/bc' },
									{ label: 'ACT', slug: 'algorithms/imitation/act' },
									{ label: 'SmolVLA', slug: 'algorithms/imitation/smolvla' },
								],
							},
							{
								label: '8 · From demonstrations',
								collapsed: true,
								items: [
									{ label: 'HIL-SERL and DAgger', slug: 'algorithms/imitation/hil-serl' },
								],
							},
							{
								label: '9 · Zero-shot',
								collapsed: true,
								items: [
									{ label: 'Forward-Backward', slug: 'algorithms/forward-backward' },
								],
							},
							{ label: 'Writing your own', slug: 'algorithms/custom' },
							],
						},
						{
							label: 'Planners',
							collapsed: true,
							items: [
								{ label: 'Overview', slug: 'planners' },
								{ label: 'Trajectory optimization', slug: 'planners/trajectory' },
								{ label: 'Tree search', slug: 'planners/tree-search' },
							],
						},
					],
				},
				{
					label: 'Robots',
					items: [
						{ label: 'Overview and safety', slug: 'robots' },
						{ label: 'SO-101 setup', slug: 'robots/so101-setup' },
						{ label: 'Teleop and recording', slug: 'robots/teleop-recording' },
						{ label: 'Cameras and calibration', slug: 'robots/cameras' },
						{ label: 'The sim twin', slug: 'robots/sim-twin' },
						{ label: 'Deployment', slug: 'robots/deployment' },
						{ label: 'Jetson Orin', slug: 'robots/jetson' },
					],
				},
				{
					label: 'Infrastructure',
					items: [
						{
							label: 'Neural networks',
							collapsed: true,
							items: [
								{ label: 'Overview', slug: 'nn' },
								{ label: 'Modules and Params', slug: 'nn/modules' },
								{ label: 'Primitives', slug: 'nn/primitives' },
								{ label: 'Combinators and models', slug: 'nn/combinators' },
								{ label: 'Optimizers, losses, init', slug: 'nn/optimizers' },
								{ label: 'Training', slug: 'nn/training' },
							],
						},
						{
							label: 'Data',
							collapsed: true,
							items: [
								{ label: 'Overview', slug: 'data' },
								{ label: 'Replay buffers', slug: 'data/replay' },
								{ label: 'Trajectory store', slug: 'data/trajectory-store' },
								{ label: 'Remote datasets', slug: 'data/remote' },
								{ label: 'LeRobot datasets', slug: 'data/lerobot' },
							],
						},
						{
							label: 'Rendering',
							collapsed: true,
							items: [
								{ label: 'Overview', slug: 'rendering' },
								{ label: 'Ray-traced cameras', slug: 'rendering/cameras' },
								{ label: 'Viewers and physics studio', slug: 'rendering/viewers' },
							],
						},
						{
							label: 'Tooling',
							collapsed: true,
							items: [
								{ label: 'Logging and metrics', slug: 'tooling/logging' },
								{ label: 'noeira cloud', slug: 'tooling/monitor' },
							],
						},
					],
				},
				{
					label: 'Project',
					items: [
						{ label: 'Toolchain', slug: 'project/toolchain' },
						{ label: 'Testing', slug: 'project/testing' },
						{ label: 'Contributing', slug: 'project/contributing' },
					],
				},
			],
		}),
	],
});
