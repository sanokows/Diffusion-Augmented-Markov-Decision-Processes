"""2-D walker with orientation-conditioned GMM action rewards.

Compared to the classic double-well env, this variant:
- allows every absolute orientation in [0, 360) degrees
- partitions orientation into `num_action_state` bins
- assigns one equal-weight Gaussian-mixture reward landscape per bin
- computes reward as `log p_GMM(a)` directly in action space for `a in [-1, 1]`
- samples Gaussian means directly in A-space
- optionally supports a point-symmetric mode:
  - each part's GMM is symmetric around `a = 0`
  - opposite heading parts (180 deg apart) share the same landscape
"""

from __future__ import annotations

from math import ceil, sqrt
from typing import Sequence

from flax import struct
import distrax
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TurningGMMState:
    """Environment state for `TurningGMMEnv`."""

    pos: jax.Array
    direction: jax.Array
    angle: jax.Array
    t: jax.Array
    rng: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


class TurningGMMEnv:
    """Orientation-conditioned GMM turning environment.

    Dynamics:
    - action `a in [-1, 1]` is mapped to a relative turn in
      `[min_turn_deg, max_turn_deg]`
    - heading is updated by that relative turn
    - agent moves forward by `step_size` along the new heading

    Reward:
    - heading determines an action-part index in `[0, num_action_state)`
    - each part has its own equal-weight Gaussian mixture in A-space
    - Gaussian means are sampled in `a in [-1 + d*sigma, 1 - d*sigma]`
    - reward is `log(p_GMM(a))`
    - optional `point_symmetric_mode` enforces:
      - per-part symmetry around `a = 0`
      - shared landscapes between opposite heading parts
    """

    def __init__(
        self,
        *,
        horizon: int = 200,
        step_size: float = 1.0,
        action_turn_range_deg: tuple[float, float] = (-100.0, 100.0),
        initial_heading_deg: float = 0.0,
        randomize_initial_heading: bool = True,
        num_action_state: int = 8,
        num_gmm_components: int = 5,
        gmm_mean_a_margin_d: float = 5,
        gmm_std: float = 0.05,
        gmm_seed: int | None = None,
        point_symmetric_mode: bool = False,
        transition_noise_deg: float = 0.0,
        action_clip_eps: float = 1e-6,
        min_density: float = 1e-30,
        mask_state_in_observation: bool = False,
        use_distance_state: bool = False,
        jit: bool = False,
    ) -> None:
        self.horizon = int(horizon)
        self.step_size = float(step_size)

        min_turn_deg, max_turn_deg = action_turn_range_deg
        self.min_turn_deg = float(min_turn_deg)
        self.max_turn_deg = float(max_turn_deg)
        self.min_turn_radians = jnp.deg2rad(jnp.asarray(self.min_turn_deg, dtype=jnp.float32))
        self.max_turn_radians = jnp.deg2rad(jnp.asarray(self.max_turn_deg, dtype=jnp.float32))

        self.initial_heading_radians = jnp.deg2rad(
            jnp.asarray(initial_heading_deg % 360.0, dtype=jnp.float32)
        )
        self.randomize_initial_heading = bool(randomize_initial_heading)

        self.num_action_state = int(num_action_state)
        self.num_gmm_components = int(num_gmm_components)
        self.gmm_mean_a_margin_d = float(gmm_mean_a_margin_d)
        self.gmm_std = float(gmm_std)
        self.gmm_mean_a_margin = self.gmm_mean_a_margin_d * self.gmm_std
        self.gmm_mean_a_min = -1.0 + self.gmm_mean_a_margin
        self.gmm_mean_a_max = 1.0 - self.gmm_mean_a_margin
        self.gmm_seed = gmm_seed
        self.point_symmetric_mode = bool(point_symmetric_mode)
        self.action_clip_eps = float(action_clip_eps)
        self.min_density = float(min_density)

        self.transition_noise_radians = jnp.deg2rad(
            jnp.asarray(abs(transition_noise_deg), dtype=jnp.float32)
        )
        self.mask_state_in_observation = bool(mask_state_in_observation)
        self.use_distance_state = bool(use_distance_state)
        self.jit = bool(jit)

        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if not self.min_turn_deg < self.max_turn_deg:
            raise ValueError("action_turn_range_deg must satisfy min < max.")
        if self.num_action_state <= 0:
            raise ValueError("num_action_state must be >= 1.")
        if self.num_gmm_components <= 0:
            raise ValueError("num_gmm_components must be >= 1.")
        if self.gmm_std <= 0.0:
            raise ValueError("gmm_std must be > 0.")
        if self.gmm_mean_a_margin_d <= 0.0:
            raise ValueError("gmm_mean_a_margin_d must be > 0.")
        if not self.gmm_mean_a_min < self.gmm_mean_a_max:
            raise ValueError(
                "gmm_mean_a_margin_d * gmm_std is too large; no valid A-space interval remains."
            )
        if self.point_symmetric_mode and (self.num_action_state % 2 != 0):
            raise ValueError(
                "point_symmetric_mode requires an even num_action_state "
                "for 180-degree action-part pairing."
            )
        if self.point_symmetric_mode and self.num_gmm_components > 1 and self.gmm_mean_a_margin > 0.5:
            raise ValueError(
                "point_symmetric_mode with num_gmm_components > 1 requires "
                "gmm_mean_a_margin_d * gmm_std <= 0.5."
            )
        if not (0.0 < self.action_clip_eps < 0.5):
            raise ValueError("action_clip_eps must be in (0, 0.5).")
        if self.min_density <= 0.0:
            raise ValueError("min_density must be > 0.")

        self.action_part_width_deg = 360.0 / float(self.num_action_state)

        mean_rng = np.random.default_rng(self.gmm_seed)
        means_a = self._sample_gmm_means_a(mean_rng)
        self._gmm_means_a = jnp.asarray(means_a, dtype=jnp.float32)
        # Backward-compatible alias; means are now defined directly in A-space.
        self._gmm_means_x = self._gmm_means_a

        self._reset_fn = jax.jit(self._reset_impl) if self.jit else self._reset_impl
        self._step_fn = jax.jit(self._step_impl) if self.jit else self._step_impl

    @property
    def action_size(self) -> int:
        return 1

    @property
    def observation_size(self) -> int:
        return 1 if self.use_distance_state else 2

    def _observation(self, pos: jax.Array, direction: jax.Array) -> jax.Array:
        if self.use_distance_state:
            dist = jnp.linalg.norm(pos, axis=-1, keepdims=True).astype(jnp.float32)
            if self.mask_state_in_observation:
                return jnp.zeros_like(dist, dtype=jnp.float32)
            return dist
        if self.mask_state_in_observation:
            return jnp.zeros_like(direction, dtype=jnp.float32)
        return jnp.asarray(direction, dtype=jnp.float32)

    def _sample_initial_heading(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        if not self.randomize_initial_heading:
            if rng.ndim == 2:
                batch = rng.shape[0]
                angle = jnp.full((batch,), self.initial_heading_radians, dtype=jnp.float32)
                return rng, angle
            angle = jnp.asarray(self.initial_heading_radians, dtype=jnp.float32)
            return rng, angle

        if rng.ndim == 2:

            def split_and_sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
                next_key, angle_key = jax.random.split(key)
                angle = jax.random.uniform(
                    angle_key,
                    (),
                    minval=0.0,
                    maxval=2.0 * jnp.pi,
                    dtype=jnp.float32,
                )
                return next_key, angle

            next_rng, angle = jax.vmap(split_and_sample)(rng)
            return next_rng, angle

        next_rng, angle_key = jax.random.split(rng)
        angle = jax.random.uniform(
            angle_key,
            (),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float32,
        )
        return next_rng, angle

    def _reset_impl(self, rng: jax.Array) -> TurningGMMState:
        rng, angle = self._sample_initial_heading(rng)
        if rng.ndim == 2:
            batch = rng.shape[0]
            pos = jnp.zeros((batch, 2), dtype=jnp.float32)
            t = jnp.zeros((batch,), dtype=jnp.int32)
            reward = jnp.zeros((batch,), dtype=jnp.float32)
            done = jnp.zeros((batch,), dtype=bool)
        else:
            pos = jnp.zeros((2,), dtype=jnp.float32)
            t = jnp.asarray(0, dtype=jnp.int32)
            reward = jnp.asarray(0.0, dtype=jnp.float32)
            done = jnp.asarray(False)

        direction = self._angle_to_direction(angle)
        info = {
            "steps": jnp.asarray(t, dtype=jnp.float32),
            "truncation": jnp.zeros_like(t, dtype=jnp.float32),
        }
        obs = self._observation(pos, direction)
        return TurningGMMState(
            pos=pos,
            direction=direction,
            angle=angle,
            t=t,
            rng=rng,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def reset(self, rng: jax.Array) -> TurningGMMState:
        return self._reset_fn(rng)

    def _step_impl(self, state: TurningGMMState, action: jax.Array) -> TurningGMMState:
        reset_state = self._reset_impl(state.rng)

        turn_action = self._extract_turn_action(action)
        action_part_idx = self._action_part_index_from_angle(state.angle)
        reward = self.reward_from_action_and_part(turn_action, action_part_idx)

        rng, noise = self._sample_transition_noise(state.rng)
        delta_angle = self._action_to_turn_radians(turn_action) + noise

        next_angle = self._wrap_angle_0_2pi(state.angle + delta_angle)
        next_direction = self._angle_to_direction(next_angle)
        next_pos = state.pos + self.step_size * next_direction
        next_t = state.t + 1
        done = next_t >= self.horizon
        info = {
            "steps": jnp.asarray(next_t, dtype=jnp.float32),
            "truncation": done.astype(jnp.float32),
        }

        stepped_state = TurningGMMState(
            pos=next_pos,
            direction=next_direction,
            angle=next_angle,
            t=next_t,
            rng=rng,
            obs=self._observation(next_pos, next_direction),
            reward=reward,
            done=done,
            info=info,
        )

        def select_reset(mask: jax.Array, reset_value: jax.Array, step_value: jax.Array) -> jax.Array:
            mask = jnp.asarray(mask, dtype=bool)
            while mask.ndim < step_value.ndim:
                mask = mask[..., None]
            return jnp.where(mask, reset_value, step_value)

        return jax.tree_util.tree_map(
            lambda r, s: select_reset(state.done, r, s),
            reset_state,
            stepped_state,
        )

    def step(self, state: TurningGMMState, action: jax.Array) -> TurningGMMState:
        return self._step_fn(state, action)

    def _sample_gmm_means_a(self, mean_rng: np.random.Generator) -> np.ndarray:
        if not self.point_symmetric_mode:
            return mean_rng.uniform(
                low=self.gmm_mean_a_min,
                high=self.gmm_mean_a_max,
                size=(self.num_action_state, self.num_gmm_components),
            ).astype(np.float32)

        half_parts = self.num_action_state // 2
        means_a = np.empty((self.num_action_state, self.num_gmm_components), dtype=np.float32)
        for part_idx in range(half_parts):
            part_means = self._sample_symmetric_part_means(mean_rng)
            means_a[part_idx] = part_means
            means_a[part_idx + half_parts] = part_means
        return means_a

    def _sample_symmetric_part_means(self, mean_rng: np.random.Generator) -> np.ndarray:
        pair_count = self.num_gmm_components // 2
        positive_means = self._sample_positive_symmetric_means(mean_rng, pair_count)

        if self.num_gmm_components % 2 == 1:
            means = np.concatenate(
                [-positive_means, np.asarray([0.0], dtype=np.float32), positive_means],
                axis=0,
            )
        else:
            means = np.concatenate([-positive_means, positive_means], axis=0)

        mean_rng.shuffle(means)
        return means.astype(np.float32)

    def _sample_positive_symmetric_means(
        self,
        mean_rng: np.random.Generator,
        count: int,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.float32)

        low = self.gmm_mean_a_margin
        high = 1.0 - self.gmm_mean_a_margin
        if low > high:
            raise ValueError("No valid positive interval for symmetric mean sampling.")
        if np.isclose(low, high):
            return np.full((count,), low, dtype=np.float32)
        return mean_rng.uniform(low=low, high=high, size=(count,)).astype(np.float32)

    def reward_from_action_and_part(self, action: jax.Array, action_part_idx: jax.Array) -> jax.Array:
        log_prob = self._gmm_log_prob_from_action_and_part(action, action_part_idx)
        return jnp.maximum(log_prob, jnp.log(jnp.asarray(self.min_density, dtype=jnp.float32)))

    def reward_from_action(
        self,
        action: jax.Array,
        *,
        reference_heading_deg: float = 0.0,
    ) -> jax.Array:
        part_idx = self.action_part_from_orientation_deg(reference_heading_deg)
        return self.reward_from_action_and_part(action, part_idx)

    def potential_from_action(
        self,
        action: jax.Array,
        *,
        reference_heading_deg: float = 0.0,
    ) -> jax.Array:
        return -self.reward_from_action(action, reference_heading_deg=reference_heading_deg)

    def reward_from_angle(
        self,
        angle: jax.Array,
        *,
        reference_heading_deg: float = 0.0,
    ) -> jax.Array:
        turn_deg = jnp.rad2deg(jnp.asarray(angle, dtype=jnp.float32))
        action = self._turn_degrees_to_action(turn_deg)
        return self.reward_from_action(action, reference_heading_deg=reference_heading_deg)

    def potential_from_angle(
        self,
        angle: jax.Array,
        *,
        reference_heading_deg: float = 0.0,
    ) -> jax.Array:
        return -self.reward_from_angle(angle, reference_heading_deg=reference_heading_deg)

    def gmm_density_from_action_and_part(self, action: jax.Array, action_part_idx: jax.Array) -> jax.Array:
        log_prob = self._gmm_log_prob_from_action_and_part(action, action_part_idx)
        return jnp.exp(log_prob)

    def _gmm_log_prob_from_action_and_part(self, action: jax.Array, action_part_idx: jax.Array) -> jax.Array:
        action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32),
            -1.0 + self.action_clip_eps,
            1.0 - self.action_clip_eps,
        )
        action_part_idx = jnp.asarray(action_part_idx, dtype=jnp.int32)
        action_part_idx = jnp.clip(action_part_idx, 0, self.num_action_state - 1)

        means = jnp.take(self._gmm_means_a, action_part_idx, axis=0)
        logits = jnp.zeros_like(means, dtype=jnp.float32)
        component_scales = jnp.full_like(means, self.gmm_std, dtype=jnp.float32)

        gmm = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits),
            components_distribution=distrax.Normal(loc=means, scale=component_scales),
        )
        return gmm.log_prob(action)

    def gmm_density_from_x_and_part(self, x: jax.Array, action_part_idx: jax.Array) -> jax.Array:
        """Backward-compatible alias: `x` is interpreted as action-space value."""
        return self.gmm_density_from_action_and_part(x, action_part_idx)

    def reward_landscape(
        self,
        num_points: int = 721,
        *,
        reference_heading_deg: float = 0.0,
        min_deg: float | None = None,
        max_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if min_deg is None:
            min_deg = self.min_turn_deg
        if max_deg is None:
            max_deg = self.max_turn_deg

        degrees = np.linspace(min_deg, max_deg, num_points, dtype=np.float32)
        radians = np.deg2rad(degrees).astype(np.float32)
        reward = np.asarray(
            self.reward_from_angle(
                jnp.asarray(radians, dtype=jnp.float32),
                reference_heading_deg=reference_heading_deg,
            )
        )
        potential = -reward
        return degrees, potential, reward

    def action_reward_profile(
        self,
        num_points: int = 401,
        *,
        reference_heading_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
        turn_degrees = np.asarray(
            self._action_to_turn_degrees(jnp.asarray(actions, dtype=jnp.float32))
        )
        rewards = np.asarray(
            self.reward_from_action(
                jnp.asarray(actions, dtype=jnp.float32),
                reference_heading_deg=reference_heading_deg,
            )
        )
        return actions, turn_degrees, rewards

    def action_potential_profile(
        self,
        num_points: int = 401,
        *,
        reference_heading_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
        turn_degrees = np.asarray(
            self._action_to_turn_degrees(jnp.asarray(actions, dtype=jnp.float32))
        )
        potential = np.asarray(
            self.potential_from_action(
                jnp.asarray(actions, dtype=jnp.float32),
                reference_heading_deg=reference_heading_deg,
            )
        )
        return actions, turn_degrees, potential

    def action_part_from_orientation_deg(self, orientation_deg: float | jax.Array) -> jax.Array:
        orientation_deg = jnp.asarray(orientation_deg, dtype=jnp.float32)
        orientation_deg = jnp.mod(orientation_deg, 360.0)
        idx = jnp.floor(orientation_deg / self.action_part_width_deg).astype(jnp.int32)
        return jnp.clip(idx, 0, self.num_action_state - 1)

    def action_part_degree_range(self, action_part_idx: int) -> tuple[float, float]:
        if not (0 <= int(action_part_idx) < self.num_action_state):
            raise ValueError(
                f"action_part_idx must be in [0, {self.num_action_state - 1}]."
            )
        start = float(action_part_idx) * self.action_part_width_deg
        end = start + self.action_part_width_deg
        return start, end

    def visualize_action_part_reward_landscape(
        self,
        action_part_idx: int,
        *,
        num_points: int = 801,
        output_path: str | None = None,
        dpi: int = 220,
    ):
        """Plot one action-part reward landscape in A-space and mapped turn space.

        Left panel:
        - reward as `log p_GMM(a)` over action `a`.

        Right panel:
        - reward over relative orientation degrees obtained from action mapping.
        """

        if not (0 <= int(action_part_idx) < self.num_action_state):
            raise ValueError(
                f"action_part_idx must be in [0, {self.num_action_state - 1}]."
            )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        part_idx = int(action_part_idx)
        part_start_deg, part_end_deg = self.action_part_degree_range(part_idx)

        means_part = np.asarray(self._gmm_means_a[part_idx], dtype=np.float32)
        a_pad = max(0.05, 6.0 * self.gmm_std)
        a_min = float(max(-1.0, np.min(means_part) - a_pad))
        a_max = float(min(1.0, np.max(means_part) + a_pad))
        a_values = np.linspace(a_min, a_max, num_points, dtype=np.float32)
        density_a = np.asarray(
            self.gmm_density_from_action_and_part(
                jnp.asarray(a_values, dtype=jnp.float32),
                jnp.asarray(part_idx, dtype=jnp.int32),
            )
        )
        reward_a_direct = np.log(np.maximum(density_a, self.min_density)).astype(np.float32)

        actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
        relative_turn_deg = np.asarray(
            self._action_to_turn_degrees(jnp.asarray(actions, dtype=jnp.float32))
        )
        reward_a = np.asarray(
            self.reward_from_action_and_part(
                jnp.asarray(actions, dtype=jnp.float32),
                jnp.asarray(part_idx, dtype=jnp.int32),
            )
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)

        axes[0].plot(a_values, reward_a_direct, color="tab:blue", linewidth=2.0)
        axes[0].set_xlabel("action a")
        axes[0].set_ylabel("log p_GMM(a)")
        axes[0].set_title("A-space reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(relative_turn_deg, reward_a, color="tab:orange", linewidth=2.0)
        axes[1].set_xlabel("Relative orientation (turn) [deg]")
        axes[1].set_ylabel("log p_GMM(a)")
        axes[1].set_title("A-space mapped to relative orientation")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Action part {part_idx}: heading in [{part_start_deg:.2f}, {part_end_deg:.2f}) deg",
            fontsize=12,
            fontweight="semibold",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

        if output_path is not None:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

        return fig

    def rollout_positions(self, states: Sequence[TurningGMMState]) -> np.ndarray:
        positions, _, _ = self._states_to_arrays(states)
        return positions

    def render_trajectory(
        self,
        trajectory: Sequence[TurningGMMState] | np.ndarray,
        rewards: np.ndarray | None = None,
        *,
        height: int = 240,
        width: int = 320,
        overlay: bool = False,
        max_envs: int | None = 9,
        titles: Sequence[str] | None = None,
        figure_title: str | None = None,
        title_fontsize: int = 16,
        show_reward: bool = True,
    ) -> list[np.ndarray]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        positions, directions, step_rewards = self._coerce_render_inputs(trajectory, rewards)
        if not bool(show_reward):
            step_rewards = None
        if positions.ndim == 2:
            positions = positions[:, None, :]
            directions = directions[:, None, :]
            if step_rewards is not None and step_rewards.ndim == 1:
                step_rewards = step_rewards[:, None]
        num_steps, num_envs = positions.shape[:2]
        if max_envs is not None:
            num_envs = min(num_envs, max_envs)
            positions = positions[:, :num_envs]
            directions = directions[:, :num_envs]
            if step_rewards is not None:
                step_rewards = step_rewards[:, :num_envs]

        x_extent = max(1.0, float(np.max(np.abs(positions[..., 0])))) + 1.5
        y_extent = max(1.0, float(np.max(np.abs(positions[..., 1])))) + 1.5
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(num_envs, 1)))

        frames: list[np.ndarray] = []
        for idx in range(num_steps):
            aspect = "auto" if overlay else "equal"
            if overlay:
                fig, axes = plt.subplots(
                    1,
                    1,
                    figsize=(width / 100.0, height / 100.0),
                    dpi=100,
                )
                axes = np.asarray([axes])
            else:
                ncols = int(ceil(sqrt(num_envs)))
                nrows = int(ceil(num_envs / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(width / 100.0 * ncols, height / 100.0 * nrows),
                    dpi=100,
                    squeeze=False,
                )
                axes = axes.reshape(-1)

            if overlay:
                ax = axes[0]
                self._draw_background(ax, x_extent, y_extent, aspect=aspect)
                for env_idx in range(num_envs):
                    color = colors[env_idx]
                    self._draw_env(
                        ax,
                        positions[:, env_idx],
                        directions[idx, env_idx],
                        idx,
                        color,
                        reward=None
                        if step_rewards is None or idx == 0
                        else step_rewards[idx - 1, env_idx],
                        title=None,
                        title_fontsize=max(10, int(0.72 * title_fontsize)),
                    )
            else:
                for env_idx, ax in enumerate(axes):
                    if env_idx >= num_envs:
                        ax.axis("off")
                        continue
                    color = colors[env_idx]
                    self._draw_background(ax, x_extent, y_extent, aspect=aspect)
                    self._draw_env(
                        ax,
                        positions[:, env_idx],
                        directions[idx, env_idx],
                        idx,
                        color,
                        reward=None
                        if step_rewards is None or idx == 0
                        else step_rewards[idx - 1, env_idx],
                        title=(
                            f"{titles[env_idx]}, t={idx}"
                            if titles is not None and env_idx < len(titles)
                            else f"env={env_idx}, t={idx}"
                        ),
                        title_fontsize=max(10, int(0.72 * title_fontsize)),
                    )

            if figure_title is not None:
                fig.suptitle(
                    f"{figure_title} | t={idx}",
                    fontsize=int(title_fontsize),
                    fontweight="bold",
                    y=0.94,
                )
                fig.tight_layout(pad=0.6, rect=(0.0, 0.0, 1.0, 0.95))
            else:
                fig.tight_layout(pad=0.6)
            fig.canvas.draw()
            buffer = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buffer[:, :, :3].copy())
            plt.close(fig)

        return frames

    def _extract_turn_action(self, action: jax.Array) -> jax.Array:
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.ndim == 0:
            return jnp.clip(action, -1.0, 1.0)
        if action.shape[-1] == 1:
            return jnp.clip(jnp.squeeze(action, axis=-1), -1.0, 1.0)
        return jnp.clip(action[..., 0], -1.0, 1.0)

    def _action_to_turn_degrees(self, action: jax.Array) -> jax.Array:
        action = jnp.asarray(action, dtype=jnp.float32)
        # Linear mapping from [-1, 1] to [min_turn_deg, max_turn_deg].
        alpha = 0.5 * (action + 1.0)
        return self.min_turn_deg + alpha * (self.max_turn_deg - self.min_turn_deg)

    def _action_to_turn_radians(self, action: jax.Array) -> jax.Array:
        return jnp.deg2rad(self._action_to_turn_degrees(action))

    def _turn_degrees_to_action(self, turn_deg: jax.Array) -> jax.Array:
        turn_deg = jnp.asarray(turn_deg, dtype=jnp.float32)
        width = self.max_turn_deg - self.min_turn_deg
        action = 2.0 * (turn_deg - self.min_turn_deg) / width - 1.0
        return jnp.clip(action, -1.0, 1.0)

    def _action_part_index_from_angle(self, angle: jax.Array) -> jax.Array:
        orientation_deg = jnp.rad2deg(self._wrap_angle_0_2pi(angle))
        idx = jnp.floor(orientation_deg / self.action_part_width_deg).astype(jnp.int32)
        return jnp.clip(idx, 0, self.num_action_state - 1)

    def _sample_transition_noise(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        if float(self.transition_noise_radians) == 0.0:
            return rng, jnp.zeros(rng.shape[:-1], dtype=jnp.float32)
        if rng.ndim == 2:

            def split_and_sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
                next_key, noise_key = jax.random.split(key)
                noise = jax.random.uniform(
                    noise_key,
                    (),
                    minval=-self.transition_noise_radians,
                    maxval=self.transition_noise_radians,
                )
                return next_key, noise

            return jax.vmap(split_and_sample)(rng)
        next_key, noise_key = jax.random.split(rng)
        noise = jax.random.uniform(
            noise_key,
            (),
            minval=-self.transition_noise_radians,
            maxval=self.transition_noise_radians,
        )
        return next_key, noise

    def _angle_to_direction(self, angle: jax.Array) -> jax.Array:
        return jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)

    def _wrap_angle_0_2pi(self, angle: jax.Array) -> jax.Array:
        return jnp.mod(jnp.asarray(angle, dtype=jnp.float32), 2.0 * jnp.pi)

    def _coerce_render_inputs(
        self,
        trajectory: Sequence[TurningGMMState] | np.ndarray,
        rewards: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if isinstance(trajectory, np.ndarray):
            positions = np.asarray(trajectory, dtype=np.float32)
            if positions.ndim not in (2, 3):
                raise ValueError("trajectory array must have shape [T, 2] or [T, B, 2].")
            directions = self._directions_from_positions(positions)
            step_rewards = None if rewards is None else np.asarray(rewards, dtype=np.float32)
            return positions, directions, step_rewards
        return self._states_to_arrays(trajectory, rewards)

    def _states_to_arrays(
        self,
        states: Sequence[TurningGMMState],
        rewards: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if len(states) == 0:
            raise ValueError("trajectory must contain at least one state.")
        positions = np.asarray([np.asarray(state.pos) for state in states], dtype=np.float32)
        directions = np.asarray([np.asarray(state.direction) for state in states], dtype=np.float32)
        if rewards is not None:
            step_rewards = np.asarray(rewards, dtype=np.float32)
        else:
            step_rewards = np.asarray(
                [np.asarray(state.reward) for state in states[1:]], dtype=np.float32
            )
            if step_rewards.size == 0:
                step_rewards = None
        return positions, directions, step_rewards

    def _directions_from_positions(self, positions: np.ndarray) -> np.ndarray:
        if positions.ndim == 2:
            diffs = np.diff(positions, axis=0, prepend=positions[:1])
        else:
            diffs = np.diff(positions, axis=0, prepend=positions[:1, ...])
        norms = np.linalg.norm(diffs, axis=-1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)
        directions = diffs / norms
        if positions.ndim == 2:
            directions[0] = np.array(
                [
                    np.cos(float(self.initial_heading_radians)),
                    np.sin(float(self.initial_heading_radians)),
                ],
                dtype=np.float32,
            )
        else:
            directions[0] = np.broadcast_to(
                np.array(
                    [
                        np.cos(float(self.initial_heading_radians)),
                        np.sin(float(self.initial_heading_radians)),
                    ],
                    dtype=np.float32,
                ),
                directions[0].shape,
            )
        return directions.astype(np.float32)

    def _draw_background(self, ax, x_extent: float, y_extent: float, *, aspect: str = "equal") -> None:
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)
        ax.set_facecolor("#f8fbff")
        ax.set_xlim(-x_extent, x_extent)
        ax.set_ylim(-y_extent, y_extent)
        ax.set_aspect(aspect)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    def _draw_env(
        self,
        ax,
        positions: np.ndarray,
        direction: np.ndarray,
        idx: int,
        color: np.ndarray,
        reward: float | np.ndarray | None,
        title: str | None,
        title_fontsize: int = 10,
    ) -> None:
        ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=1.2, alpha=0.2)
        ax.plot(positions[: idx + 1, 0], positions[: idx + 1, 1], color=color, linewidth=2.6)
        ax.scatter(
            positions[idx, 0],
            positions[idx, 1],
            color="#d62728",
            s=36,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.quiver(
            positions[idx, 0],
            positions[idx, 1],
            direction[0],
            direction[1],
            angles="xy",
            scale_units="xy",
            scale=0.6,
            color="black",
            width=0.01,
            zorder=4,
        )
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight="semibold")
        if reward is not None:
            reward_value = float(np.asarray(reward))
            ax.text(
                0.02,
                0.98,
                f"reward={reward_value:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
