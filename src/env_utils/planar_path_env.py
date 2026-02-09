"""2D planar path environment with two diverging targets."""

from __future__ import annotations

from typing import Iterable, Sequence

from flax import struct
from flax.core import frozen_dict
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class PlanarPathState:
    pos: jax.Array
    t: jax.Array
    target_dirs: jax.Array
    target_mask: jax.Array
    rng: jax.Array
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


class PlanarPathEnv:
    def __init__(
        self,
        *,
        sigma: float = 2.,
        weights: Sequence[float] = (1.0, 1.0),
        split_step: int = 5,
        rep_split_steps: int = 10,
        n_splits: int = 5,
        step_size: float = 1.0,
        eps: float = 0.2,
    ) -> None:
        self.sigma = float(sigma)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.n_splits = int(n_splits)
        self.horizon = 1000
        rep = max(1, self.horizon // (self.n_splits + 1))
        self.split_step = rep
        self.rep_split_steps = rep
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.rep_split_steps = self.rep_split_steps if self.rep_split_steps is not None else int(1e9)
        self.n_splits = max(self.n_splits, 1)
        self.max_branches = 2 ** self.n_splits
        bits = np.arange(self.max_branches, dtype=np.int32)[:, None]
        shifts = np.arange(self.n_splits, dtype=np.int32)[None, :]
        branch_bits = (bits >> shifts) & 1
        self._branch_bits = jnp.asarray(branch_bits)
        self._branch_signs = jnp.where(self._branch_bits == 1, 1.0, -1.0)
        self._branch_signs_np = np.where(branch_bits == 1, 1.0, -1.0).astype(np.float32)
        base_weights = np.asarray(weights, dtype=np.float32)
        if base_weights.size == 1:
            full_weights = np.repeat(base_weights, self.max_branches)
        elif base_weights.size == self.max_branches:
            full_weights = base_weights
        else:
            reps = int(np.ceil(self.max_branches / base_weights.size))
            full_weights = np.tile(base_weights, reps)[: self.max_branches]
        self._weights_full = jnp.asarray(full_weights)

    @property
    def action_size(self) -> int:
        return 2

    @property
    def observation_size(self) -> int:
        return (self.max_branches * 2) + self.max_branches

    def reset(self, rng: jax.Array) -> PlanarPathState:
        if rng.ndim == 2:
            batch = rng.shape[0]
            pos = jnp.zeros((batch, 2), dtype=jnp.float32)
            t = jnp.zeros((batch,), dtype=jnp.int32)
        else:
            pos = jnp.zeros((2,), dtype=jnp.float32)
            t = jnp.array(0)
        target_dirs = self._target_dirs(pos, t)
        target_mask = self._active_mask(t)
        target_dirs, target_mask = self._canonicalize_targets(target_dirs, target_mask)
        obs = self._build_obs(target_dirs, target_mask)
        reward = jnp.array(0.0, dtype=jnp.float32)
        done = jnp.array(False)
        if t.ndim > 0:
            reward = jnp.zeros_like(t, dtype=jnp.float32)
            done = jnp.zeros_like(t, dtype=bool)
        info = dict(
            {"steps": jnp.array(t, dtype=jnp.float32), "truncation": jnp.zeros_like(t, dtype=jnp.float32)}
        )
        return PlanarPathState(
            pos=pos,
            t=t,
            target_dirs=target_dirs,
            target_mask=target_mask,
            rng=rng,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def step(self, state: PlanarPathState, action: jax.Array) -> PlanarPathState:
        if action.ndim == 1:
            angle_action = jnp.clip(action[0], -1.0, 1.0)
            size_action = jnp.clip(action[1], -1.0, 1.0) if action.shape[0] > 1 else 0.0
        else:
            angle_action = jnp.clip(action[..., 0], -1.0, 1.0)
            size_action = jnp.clip(action[..., 1], -1.0, 1.0)
        angle = angle_action * jnp.pi
        if state.rng.ndim == 2:
            def split_and_noise(k):
                k1, k2 = jax.random.split(k)
                n = jax.random.uniform(k2, (), minval=-self.eps, maxval=self.eps)
                return k1, n

            rng, noise = jax.vmap(split_and_noise)(state.rng)
        else:
            rng, noise_key = jax.random.split(state.rng)
            noise = jax.random.uniform(noise_key, (), minval=-self.eps, maxval=self.eps)
        step_size = self.step_size + self.eps * size_action + noise
        direction = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
        delta = step_size[..., None] * direction
        next_pos = state.pos + delta
        next_t = state.t + 1

        targets = self._targets(next_t)
        mask = self._active_mask(next_t).astype(jnp.float32)
        diffs = next_pos[None, :] - targets
        dist_sq = jnp.sum(diffs * diffs, axis=-1)
        if mask.ndim == 2 and mask.shape[0] == next_pos.shape[0]:
            mask_for_reward = mask.T
        else:
            mask_for_reward = mask
        if mask_for_reward.ndim == 2:
            masked_dist_sq = jnp.where(mask_for_reward > 0.0, dist_sq, jnp.inf)
            min_dist_sq = jnp.min(masked_dist_sq, axis=0)
        else:
            masked_dist_sq = jnp.where(mask_for_reward > 0.0, dist_sq, jnp.inf)
            min_dist_sq = jnp.min(masked_dist_sq)
        reward = jnp.exp(-min_dist_sq / self.sigma)
        done = next_t >= self.horizon
        next_dirs = self._target_dirs(next_pos, next_t)
        next_mask = self._active_mask(next_t)
        next_dirs, next_mask = self._canonicalize_targets(next_dirs, next_mask)
        obs = self._build_obs(next_dirs, next_mask)
        info = dict(
            {"steps": jnp.array(next_t, dtype=jnp.float32), "truncation": jnp.zeros_like(next_t, dtype=jnp.float32)}
        )
        next_state = PlanarPathState(
            pos=next_pos,
            t=next_t,
            target_dirs=next_dirs,
            target_mask=next_mask,
            rng=rng,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )
        return next_state

    def _targets(self, t: jax.Array) -> jax.Array:
        t = t.astype(jnp.float32)
        base = jnp.stack([t, jnp.zeros_like(t)], axis=-1)
        split = jnp.array(self.split_step, dtype=jnp.float32)
        t_rel = jnp.maximum(t - split, 0.0)
        rep = jnp.array(self.rep_split_steps, dtype=jnp.float32)

        lengths = []
        for i in range(self.n_splits):
            seg_start = split + jnp.array(i, dtype=jnp.float32) * rep
            seg_len = jnp.clip(t - seg_start, 0.0, rep)
            lengths.append(seg_len)
        lengths = jnp.stack(lengths, axis=0)

        l0 = jnp.minimum(t, split)
        diag_scale = lengths / jnp.sqrt(2.0)
        total_diag = jnp.sum(diag_scale, axis=0)

        x = l0 + total_diag
        y = jnp.matmul(self._branch_signs, diag_scale)
        x_broadcast = jnp.broadcast_to(x, y.shape)
        targets = jnp.stack([x_broadcast, y], axis=-1)

        use_diag = t >= self.split_step
        if t.ndim > 0:
            targets = jnp.where(use_diag[None, :, None], targets, base[None, :, :])
        else:
            targets = jnp.where(use_diag, targets, base[None, :])
        return targets

    def _canonicalize_targets(
        self, target_dirs: jax.Array, target_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if target_dirs.ndim == 3 and target_dirs.shape[0] == self.max_branches:
            target_dirs = jnp.swapaxes(target_dirs, 0, 1)
        if target_mask.ndim == 2 and target_mask.shape[0] == self.max_branches:
            target_mask = jnp.swapaxes(target_mask, 0, 1)
        return target_dirs, target_mask

    def _build_obs(self, target_dirs: jax.Array, target_mask: jax.Array) -> jax.Array:
        if target_dirs.ndim == 3:
            flat_dirs = target_dirs.reshape(target_dirs.shape[0], -1)
            return jnp.concatenate([flat_dirs, target_mask], axis=-1)
        flat_dirs = target_dirs.reshape(-1)
        return jnp.concatenate([flat_dirs, target_mask], axis=0)

    def _active_mask(self, t: jax.Array) -> jax.Array:
        t = t.astype(jnp.float32)
        split = jnp.array(self.split_step, dtype=jnp.float32)
        rep = jnp.array(self.rep_split_steps, dtype=jnp.float32)
        t_rel = t - split
        num_splits = jnp.where(
            t < split,
            0.0,
            1.0 + jnp.floor_divide(jnp.maximum(t_rel, 0.0), rep),
        )
        num_splits = jnp.minimum(num_splits, jnp.array(self.n_splits, dtype=jnp.float32))
        split_idx = num_splits.astype(jnp.int32)
        bit_indices = jnp.arange(self.n_splits)
        if t.ndim > 0:
            keep = (
                bit_indices[None, None, :] < split_idx[:, None, None]
            ) | (self._branch_bits[None, :, :] == 0)
            return jnp.all(keep, axis=-1)
        keep = (bit_indices < split_idx) | (self._branch_bits == 0)
        return jnp.all(keep, axis=-1)

    def _target_dirs(self, pos: jax.Array, t: jax.Array) -> jax.Array:
        targets = self._targets(t)
        diffs = targets - pos[None, :]
        norms = jnp.linalg.norm(diffs, axis=-1, keepdims=True)
        norms = jnp.where(norms > 1e-6, norms, 1.0)
        clipped = jnp.minimum(norms, 2.0)
        return diffs / norms * clipped

    def target_trajectory(self, steps: int) -> np.ndarray:
        times = np.arange(steps + 1, dtype=np.float32)
        base = np.stack([times, np.zeros_like(times)], axis=-1)
        split = float(self.split_step)
        rep = float(self.rep_split_steps)

        lengths = []
        for i in range(self.n_splits):
            seg_start = split + i * rep
            seg_len = np.clip(times - seg_start, 0.0, rep)
            lengths.append(seg_len)
        lengths = np.stack(lengths, axis=0)

        l0 = np.minimum(times, split)
        diag_scale = lengths / np.sqrt(2.0)
        total_diag = np.sum(diag_scale, axis=0)
        x = l0 + total_diag
        y = self._branch_signs_np @ diag_scale
        targets = np.stack([np.broadcast_to(x, y.shape), y], axis=-1)
        mask = times >= self.split_step
        targets = np.where(mask[None, :, None], targets, base[None, :, :])
        return targets

    def render_trajectory(
        self,
        positions: np.ndarray,
        rewards: np.ndarray | None = None,
        *,
        height: int = 240,
        width: int = 320,
    ) -> list[np.ndarray]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = positions.shape[0] - 1
        targets = self.target_trajectory(steps)
        split_times = [
            self.split_step + i * self.rep_split_steps for i in range(self.n_splits)
        ]
        split_points = []
        for split_t in split_times:
            split_t = int(split_t)
            split_targets = self._targets_np(split_t)
            split_mask = self._active_mask_np(split_t, split_targets.shape[0])
            split_points.append(split_targets[split_mask])
        if split_points:
            split_points = np.concatenate(split_points, axis=0)
        else:
            split_points = np.zeros((0, 2), dtype=np.float32)

        x_max = max(
            float(np.max(targets[..., 0])),
            float(np.max(positions[:, 0])),
            float(np.max(split_points[:, 0])) if split_points.size else 0.0,
        ) + 1.0
        y_max = max(
            float(np.max(np.abs(targets[..., 1]))),
            float(np.max(np.abs(positions[:, 1]))),
            float(np.max(np.abs(split_points[:, 1]))) if split_points.size else 0.0,
        ) + 1.0

        frames: list[np.ndarray] = []
        for idx in range(positions.shape[0]):
            fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
            for b in range(targets.shape[0]):
                ax.plot(
                    targets[b, :, 0],
                    targets[b, :, 1],
                    "--",
                    color="tab:blue" if b % 2 == 0 else "tab:orange",
                    alpha=0.25,
                )
            if self.rep_split_steps is not None and self.n_splits > 1:
                rep = self.rep_split_steps
                for split_idx, split_t in enumerate(split_times):
                    if split_t >= steps:
                        continue
                    end_t = min(split_t + rep, steps)
                    local = np.arange(0, end_t - split_t + 1, dtype=np.float32)
                    diag = local / np.sqrt(2.0)
                    origin_up = targets[0, split_t]
                    origin_down = targets[1, split_t]
                    ax.plot(
                        origin_up[0] + diag,
                        origin_up[1] + diag,
                        color="tab:blue",
                        linewidth=0.8,
                        alpha=0.5,
                    )
                    ax.plot(
                        origin_down[0] + diag,
                        origin_down[1] - diag,
                        color="tab:orange",
                        linewidth=0.8,
                        alpha=0.5,
                    )
            ax.plot(positions[:, 0], positions[:, 1], color="black", linewidth=1, alpha=0.4, label="agent path")
            ax.plot(positions[: idx + 1, 0], positions[: idx + 1, 1], color="black", linewidth=2, label="agent")
            ax.scatter(positions[idx, 0], positions[idx, 1], color="red", s=40, zorder=3)
            if targets.shape[0] > 0:
                active_mask = self._active_mask_np(idx, targets.shape[0])
                ax.scatter(
                    targets[active_mask, idx, 0],
                    targets[active_mask, idx, 1],
                    color="tab:blue",
                    s=20,
                    zorder=3,
                )
            if split_times:
                for split_t in split_times:
                    split_t = int(split_t)
                    split_targets = self._targets_np(split_t)
                    split_mask = self._active_mask_np(split_t, split_targets.shape[0])
                    ax.scatter(
                        split_targets[split_mask, 0],
                        split_targets[split_mask, 1],
                        color="gray",
                        marker="s",
                        s=14,
                        alpha=0.6,
                        zorder=2,
                    )

            pos = positions[idx]
            active_mask = self._active_mask_np(idx, targets.shape[0])
            dirs = targets[:, idx, :] - pos[None, :]
            norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
            norms = np.where(norms > 1e-6, norms, 1.0)
            dirs = 3.5 * (dirs / norms)
            if np.any(active_mask):
                ax.quiver(
                    np.full(active_mask.sum(), pos[0]),
                    np.full(active_mask.sum(), pos[1]),
                    dirs[active_mask, 0],
                    dirs[active_mask, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color="tab:green",
                    width=0.007,
                    headwidth=4.5,
                    headlength=6.0,
                    headaxislength=5.5,
                    zorder=4,
                )
            ax.set_xlim(-1.0, x_max)
            ax.set_ylim(-y_max, y_max)
            ax.set_aspect("equal")
            ax.set_title(f"t={idx}")
            ax.legend(loc="upper left", fontsize=6)
            ax.grid(True, linewidth=0.3)
            if rewards is not None:
                reward_val = rewards[idx - 1] if idx > 0 else 0.0
                ax.text(
                    0.02,
                    0.98,
                    f"reward={reward_val:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
            fig.tight_layout(pad=0.5)

            fig.canvas.draw()
            buffer = np.asarray(fig.canvas.buffer_rgba())
            frame = buffer[:, :, :3].copy()
            frames.append(frame)
            plt.close(fig)

        return frames

    def _active_mask_np(self, t: int, num_branches: int) -> np.ndarray:
        if t < self.split_step:
            splits = 0
        else:
            splits = 1 + int(max(t - self.split_step, 0) // self.rep_split_steps)
        splits = min(splits, self.n_splits)
        bits = np.arange(num_branches)[:, None] >> np.arange(self.n_splits)[None, :]
        mask = np.all((np.arange(self.n_splits)[None, :] < splits) | ((bits & 1) == 0), axis=-1)
        return mask

    def _targets_np(self, t: int) -> np.ndarray:
        split = float(self.split_step)
        rep = float(self.rep_split_steps)
        t_rel = max(t - split, 0.0)
        lengths = []
        for i in range(self.n_splits):
            seg_start = split + i * rep
            seg_len = np.clip(t - seg_start, 0.0, rep)
            lengths.append(seg_len)
        lengths = np.asarray(lengths, dtype=np.float32)
        l0 = min(t, split)
        diag_scale = lengths / np.sqrt(2.0)
        total_diag = np.sum(diag_scale)
        x = l0 + total_diag
        y = self._branch_signs_np @ diag_scale
        targets = np.stack([np.full_like(y, x), y], axis=-1)
        if t < self.split_step:
            base = np.array([t, 0.0], dtype=np.float32)
            targets = np.tile(base[None, :], (self.max_branches, 1))
        return targets
