import optax


def get_learning_rate_scheduler(cfg, step_size):
    """Creates learning rate schedule."""
    if cfg['warmup'] == 'linear':
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=step_size,
            transition_steps=cfg['warmup_iters'])
        """Creates learning rate schedule."""
    elif cfg['warmup'] == 'const':
        warmup_fn = optax.constant_schedule(step_size)
    else:
        raise ValueError(f"No warmup scheme called {cfg['warmup']}")

    cosine_epochs = max(cfg['iters'] - cfg['warmup_iters'], 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=step_size,
        decay_steps=cosine_epochs)

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[cfg['warmup_iters']])
    return schedule_fn


def get_learning_rate_scheduler_with_end(cfg, init_lr, end_lr):
    """Creates learning rate schedule with custom end value for cosine decay."""
    if cfg['warmup'] == 'linear':
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=init_lr,
            transition_steps=cfg['warmup_iters'])
    elif cfg['warmup'] == 'const':
        warmup_fn = optax.constant_schedule(init_lr)
    else:
        raise ValueError(f"No warmup scheme called {cfg['warmup']}")

    cosine_epochs = max(cfg['iters'] - cfg['warmup_iters'], 1)
    
    # Custom cosine decay from init_lr to end_lr
    def cosine_decay_with_end(step):
        import jax.numpy as jnp
        # Normalize step to [0, 1] range
        progress = step / cosine_epochs
        progress = jnp.clip(progress, 0.0, 1.0)
        # Cosine decay formula: end_lr + 0.5 * (init_lr - end_lr) * (1 + cos(π * progress))
        decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return end_lr + (init_lr - end_lr) * decay_factor
    
    cosine_fn = cosine_decay_with_end

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[cfg['warmup_iters']])
    return schedule_fn
