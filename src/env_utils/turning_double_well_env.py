"""2-D walker with relative turning actions and a double-well turn reward."""

from __future__ import annotations

from fractions import Fraction
from functools import partial
from math import ceil, gcd, sqrt
from typing import Sequence

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TurningDoubleWellState:
    pos: jax.Array
    direction: jax.Array
    angle: jax.Array
    t: jax.Array
    rng: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


class TurningDoubleWellEnv:
    def __init__(
        self,
        *,
        horizon: int = 200,
        step_size: float = 1.0,
        max_turn_deg: float = 90.0,
        initial_heading_deg: float = 0.0,
        randomize_initial_heading: bool = True,
        well_angle_deg: float = 45.0,
        well_height: float = 0.85,
        transition_noise_deg: float = 0.,
        snap_action_to_optimal: bool = True,
        mask_state_in_observation: bool = False,
        use_distance_state: bool = False,
        jit: bool = False,
    ) -> None:
        self.horizon = int(horizon)
        self.step_size = float(step_size)
        self.max_turn_deg = float(max_turn_deg)
        self.max_turn_radians = jnp.deg2rad(jnp.asarray(max_turn_deg, dtype=jnp.float32))
        self.initial_heading_radians = jnp.deg2rad(
            jnp.asarray(initial_heading_deg, dtype=jnp.float32)
        )
        self.randomize_initial_heading = bool(randomize_initial_heading)
        self.well_angle_deg = float(abs(well_angle_deg))
        self.well_angle_radians = jnp.deg2rad(
            jnp.asarray(abs(well_angle_deg), dtype=jnp.float32)
        )
        self.well_height = float(well_height)
        self.transition_noise_radians = jnp.deg2rad(
            jnp.asarray(abs(transition_noise_deg), dtype=jnp.float32)
        )
        self.snap_action_to_optimal = bool(snap_action_to_optimal)
        self.mask_state_in_observation = bool(mask_state_in_observation)
        # If True, the observation/state exposed to the agent is the (scalar) distance to origin
        # instead of the 2-D orientation vector.
        self.use_distance_state = bool(use_distance_state)
        self.jit = bool(jit)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if not (0.0 <= self.well_height <= 1.0):
            raise ValueError("well_height must be in [0, 1].")
        if not (0.0 < self.well_angle_deg < 90.0):
            raise ValueError("well_angle_deg must be in (0, 90).")
        self._peak_u = jnp.asarray((self.well_angle_deg / 90.0) ** 2, dtype=jnp.float32)

        # Discrete set of initial headings used when both:
        # - randomize_initial_heading=True
        # - snap_action_to_optimal=True
        # The set covers all unique angles of the form k * well_angle_deg (mod 360).
        well_angle_frac = Fraction(self.well_angle_deg).limit_denominator(3600)
        denom = int(well_angle_frac.denominator)
        numer = int(well_angle_frac.numerator)
        period = (360 * denom) // gcd(numer, 360 * denom)
        if period <= 0:
            period = 1
        if period > 8192:
            raise ValueError(
                "well_angle_deg yields too many discrete initial headings "
                f"({period}); increase well_angle_deg."
            )
        heading_degrees = (np.arange(period, dtype=np.float64) * (numer / denom)).astype(np.float32)
        self._snapped_initial_heading_support_radians = jnp.deg2rad(
            jnp.asarray(heading_degrees, dtype=jnp.float32)
        )

        # JIT wrappers (useful when stepping this env from Python without an outer jit).
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
        """Return (next_rng, angle) for the initial heading."""
        if not self.randomize_initial_heading:
            if rng.ndim == 2:
                batch = rng.shape[0]
                angle = jnp.full((batch,), self.initial_heading_radians, dtype=jnp.float32)
                return rng, angle
            angle = jnp.asarray(self.initial_heading_radians, dtype=jnp.float32)
            return rng, angle

        if self.snap_action_to_optimal:
            allowed_angles = self._snapped_initial_heading_support_radians

            if rng.ndim == 2:
                def split_and_sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
                    next_key, angle_key = jax.random.split(key)
                    idx = jax.random.randint(angle_key, (), 0, allowed_angles.shape[0])
                    angle = self._wrap_angle(allowed_angles[idx])
                    return next_key, angle

                next_rng, angle = jax.vmap(split_and_sample)(rng)
                return next_rng, angle

            next_rng, angle_key = jax.random.split(rng)
            idx = jax.random.randint(angle_key, (), 0, allowed_angles.shape[0])
            angle = self._wrap_angle(allowed_angles[idx])
            return next_rng, angle

        if rng.ndim == 2:
            def split_and_sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
                next_key, angle_key = jax.random.split(key)
                angle = jax.random.uniform(
                    angle_key,
                    (),
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                    dtype=jnp.float32,
                )
                return next_key, angle

            next_rng, angle = jax.vmap(split_and_sample)(rng)
            return next_rng, angle

        next_rng, angle_key = jax.random.split(rng)
        angle = jax.random.uniform(
            angle_key,
            (),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=jnp.float32,
        )
        return next_rng, angle

    def _reset_impl(self, rng: jax.Array) -> TurningDoubleWellState:
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
        #jax.debug.print("print info steps and truncation", steps=info["steps"], truncation=info["truncation"])
        obs = self._observation(pos, direction)
        return TurningDoubleWellState(
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

    def reset(self, rng: jax.Array) -> TurningDoubleWellState:
        return self._reset_fn(rng)

    def _step_impl(self, state: TurningDoubleWellState, action: jax.Array) -> TurningDoubleWellState:
        # Auto-reset on the step *after* termination: if the caller provides a terminal
        # state, return a fresh reset state with done=False/truncation=0.
        reset_state = self._reset_impl(state.rng)

        turn_action = self._extract_turn_action(action)
        rng, noise = self._sample_transition_noise(state.rng)

        delta_angle_raw = turn_action * self.max_turn_radians + noise
        # Reward depends only on the raw turn delta (not on the absolute orientation).
        reward = self.reward_from_angle(delta_angle_raw)

        if self.snap_action_to_optimal:
            sign = jnp.where(turn_action >= 0.0, 1.0, -1.0).astype(jnp.float32)
            inv_max_turn = jnp.where(self.max_turn_radians > 0.0, 1.0 / self.max_turn_radians, 0.0)
            ratio = (self.well_angle_radians * inv_max_turn).astype(jnp.float32)
            ratio = jnp.minimum(ratio, jnp.asarray(1.0, dtype=jnp.float32))
            snapped_turn_action = jnp.clip(sign * ratio, -1.0, 1.0)
            delta_angle_transition = snapped_turn_action * self.max_turn_radians + noise
        else:
            delta_angle_transition = delta_angle_raw

        next_angle = self._wrap_angle(state.angle + delta_angle_transition)
        next_direction = self._angle_to_direction(next_angle)
        next_pos = state.pos + self.step_size * next_direction
        next_t = state.t + 1
        done = next_t >= self.horizon
        info = {
            "steps": jnp.asarray(next_t, dtype=jnp.float32),
            "truncation": done.astype(jnp.float32),
        }
        stepped_state = TurningDoubleWellState(
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

    def step(self, state: TurningDoubleWellState, action: jax.Array) -> TurningDoubleWellState:
        return self._step_fn(state, action)

    def potential_from_angle(self, angle: jax.Array) -> jax.Array:
        return 1.0 - self.reward_from_angle(angle)

    def reward_from_angle(self, angle: jax.Array) -> jax.Array:
        """Reward profile for a signed angle delta (radians)."""
        wrapped_deg = jnp.rad2deg(self._wrap_angle(angle))
        clamped_deg = jnp.clip(wrapped_deg, -90.0, 90.0)
        u = jnp.square(clamped_deg / 90.0)

        # Piecewise polynomial profile in u=(theta/90)^2:
        # - from 0 to peak_u: monotonic increase from well_height to 1
        # - from peak_u to 1: monotonic decrease from 1 to 0
        peak_u = self._peak_u
        left_s = jnp.clip(u / peak_u, 0.0, 1.0)
        right_t = jnp.clip((u - peak_u) / (1.0 - peak_u), 0.0, 1.0)

        left_reward = self.well_height + (1.0 - self.well_height) * (1.0 - (1.0 - left_s) ** 2)
        right_reward = 1.0 - right_t**2
        return jnp.where(u <= peak_u, left_reward, right_reward)

    def reward_landscape(
        self,
        num_points: int = 721,
        *,
        min_deg: float | None = None,
        max_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if min_deg is None:
            min_deg = -float(np.rad2deg(np.asarray(self.max_turn_radians)))
        if max_deg is None:
            max_deg = float(np.rad2deg(np.asarray(self.max_turn_radians)))
        degrees = np.linspace(min_deg, max_deg, num_points, dtype=np.float32)
        radians = np.deg2rad(degrees).astype(np.float32)
        potential = np.asarray(self.potential_from_angle(jnp.asarray(radians)))
        reward = np.asarray(self.reward_from_angle(jnp.asarray(radians)))
        return degrees, potential, reward

    def action_potential_profile(
        self,
        num_points: int = 401,
        *,
        reference_heading_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
        turn_degrees = actions * float(np.rad2deg(np.asarray(self.max_turn_radians)))
        # Reward/potential depend only on the turn delta; reference heading is ignored.
        heading_degrees = turn_degrees
        radians = np.deg2rad(turn_degrees).astype(np.float32)
        potential = np.asarray(self.potential_from_angle(jnp.asarray(radians)))
        return actions, heading_degrees, potential

    def action_reward_profile(
        self,
        num_points: int = 401,
        *,
        reference_heading_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)
        turn_degrees = actions * self.max_turn_deg
        # Reward depends only on the turn delta; reference heading is ignored.
        heading_degrees = turn_degrees
        radians = np.deg2rad(turn_degrees).astype(np.float32)
        rewards = np.asarray(self.reward_from_angle(jnp.asarray(radians)))
        return actions, heading_degrees, rewards

    def rollout_positions(self, states: Sequence[TurningDoubleWellState]) -> np.ndarray:
        positions, _, _ = self._states_to_arrays(states)
        return positions

    def render_trajectory(
        self,
        trajectory: Sequence[TurningDoubleWellState] | np.ndarray,
        rewards: np.ndarray | None = None,
        *,
        height: int = 240,
        width: int = 320,
        overlay: bool = False,
        show_preferred_directions: bool = False,
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

        preferred = np.asarray(
            [
                [np.cos(float(self.well_angle_radians)), np.sin(float(self.well_angle_radians))],
                [np.cos(float(self.well_angle_radians)), -np.sin(float(self.well_angle_radians))],
            ],
            dtype=np.float32,
        )

        x_extent = max(1.0, float(np.max(np.abs(positions[..., 0])))) + 1.5
        y_extent = max(1.0, float(np.max(np.abs(positions[..., 1])))) + 1.5
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(num_envs, 1)))

        frames: list[np.ndarray] = []
        for idx in range(num_steps):
            # With `aspect="equal"`, very long trajectories in x vs y can make the axes extremely
            # short (visually "squeezed" in y). For overlay renders we prefer a readable figure.
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
                self._draw_background(
                    ax,
                    x_extent,
                    y_extent,
                    preferred,
                    aspect=aspect,
                    show_preferred_directions=show_preferred_directions,
                )
                for env_idx in range(num_envs):
                    color = colors[env_idx]
                    self._draw_env(
                        ax,
                        positions[:, env_idx],
                        directions[idx, env_idx],
                        idx,
                        color,
                        reward=None if step_rewards is None or idx == 0 else step_rewards[idx - 1, env_idx],
                        title=None,
                        title_fontsize=max(10, int(0.72 * title_fontsize)),
                    )
            else:
                for env_idx, ax in enumerate(axes):
                    if env_idx >= num_envs:
                        ax.axis("off")
                        continue
                    color = colors[env_idx]
                    self._draw_background(
                        ax,
                        x_extent,
                        y_extent,
                        preferred,
                        aspect=aspect,
                        show_preferred_directions=show_preferred_directions,
                    )
                    self._draw_env(
                        ax,
                        positions[:, env_idx],
                        directions[idx, env_idx],
                        idx,
                        color,
                        reward=None if step_rewards is None or idx == 0 else step_rewards[idx - 1, env_idx],
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

    def _wrap_angle(self, angle: jax.Array) -> jax.Array:
        return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))

    def _coerce_render_inputs(
        self,
        trajectory: Sequence[TurningDoubleWellState] | np.ndarray,
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
        states: Sequence[TurningDoubleWellState],
        rewards: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if len(states) == 0:
            raise ValueError("trajectory must contain at least one state.")
        positions = np.asarray([np.asarray(state.pos) for state in states], dtype=np.float32)
        directions = np.asarray([np.asarray(state.direction) for state in states], dtype=np.float32)
        if rewards is not None:
            step_rewards = np.asarray(rewards, dtype=np.float32)
        else:
            step_rewards = np.asarray([np.asarray(state.reward) for state in states[1:]], dtype=np.float32)
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
            directions[0] = np.array([np.cos(float(self.initial_heading_radians)), np.sin(float(self.initial_heading_radians))])
        else:
            directions[0] = np.broadcast_to(
                np.array(
                    [np.cos(float(self.initial_heading_radians)), np.sin(float(self.initial_heading_radians))],
                    dtype=np.float32,
                ),
                directions[0].shape,
            )
        return directions.astype(np.float32)

    def _draw_background(
        self,
        ax,
        x_extent: float,
        y_extent: float,
        preferred: np.ndarray,
        *,
        aspect: str = "equal",
        show_preferred_directions: bool = False,
    ) -> None:
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)
        ax.set_facecolor("#f8fbff")
        if show_preferred_directions:
            ray_length = max(x_extent, y_extent)
            for vec in preferred:
                ax.plot(
                    [0.0, ray_length * vec[0]],
                    [0.0, ray_length * vec[1]],
                    "--",
                    color="tab:green",
                    linewidth=1.0,
                    alpha=0.5,
                )
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
