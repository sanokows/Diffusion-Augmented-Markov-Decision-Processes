import functools

import flax.struct as struct
import jax
import jax.numpy as jnp


class NormalizationState(struct.PyTreeNode):
    mean: struct.PyTreeNode
    var: struct.PyTreeNode
    count: int


class Normalizer:
    @functools.partial(jax.jit, static_argnums=0)
    def init(self, tree: struct.PyTreeNode) -> NormalizationState:
        return NormalizationState(
            mean=jax.tree.map(lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype), tree),
            var=jax.tree.map(lambda x: jnp.ones(x.shape[1:], dtype=x.dtype), tree),
            count=0,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> NormalizationState:
        var = jax.tree.map(lambda x: jnp.var(x, axis=0), tree)
        mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), tree)
        batch_size = jax.tree.reduce(lambda x, y: y.shape[0], tree, 0)
        delta = mean - state.mean
        count = state.count + batch_size
        new_mean = state.mean + delta * batch_size / count
        m_a = state.var * state.count
        m_b = var * batch_size
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_size / count

        return state.replace(mean=new_mean, var=M2 / count, count=count)

    @functools.partial(jax.jit, static_argnums=0)
    def normalize(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> struct.PyTreeNode:
        return jax.tree.map(
            lambda x, m, v: (x - m) / jnp.sqrt(v + 1e-8), tree, state.mean, state.var
        )


class DictNormalizationState(struct.PyTreeNode):
    obs_state: NormalizationState | None
    actions_state: NormalizationState | None


class DictNormalizer:
    """Normalizer that only touches `orig_obs` and `normed_actions` entries in a dict tree."""

    def __init__(self):
        self.obs_normalizer = Normalizer()
        self.actions_normalizer = Normalizer()

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, tree: dict) -> DictNormalizationState:
        obs_state = None
        actions_state = None
        if "orig_obs" in tree:
            obs_state = self.obs_normalizer.init(tree["orig_obs"])
        if "normed_actions" in tree:
            actions_state = self.actions_normalizer.init(tree["normed_actions"])
        return DictNormalizationState(obs_state=obs_state, actions_state=actions_state)

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: DictNormalizationState, tree: dict) -> DictNormalizationState:
        obs_state = state.obs_state
        actions_state = state.actions_state
        if obs_state is not None and "orig_obs" in tree:
            obs_state = self.obs_normalizer.update(obs_state, tree["orig_obs"])
        if actions_state is not None and "normed_actions" in tree:
            actions_state = self.actions_normalizer.update(actions_state, tree["normed_actions"])
        return state.replace(obs_state=obs_state, actions_state=actions_state)

    @functools.partial(jax.jit, static_argnums=0)
    def normalize(self, state: DictNormalizationState, tree: dict) -> dict:
        out = dict(tree)
        if state.obs_state is not None and "orig_obs" in out:
            out["orig_obs"] = self.obs_normalizer.normalize(state.obs_state, out["orig_obs"])
        if state.actions_state is not None and "normed_actions" in out:
            out["normed_actions"] = self.actions_normalizer.normalize(
                state.actions_state, out["normed_actions"]
            )
        return out
