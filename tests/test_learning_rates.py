from src.jaxrl.reppo_helpers.learning_rates import resolve_special_lr


def test_resolve_special_lr_uses_base_when_special_lr_is_none():
    assert resolve_special_lr(3e-4, None) == 3e-4


def test_resolve_special_lr_uses_explicit_special_lr():
    assert resolve_special_lr(3e-4, 1e-4) == 1e-4


def test_resolve_special_lr_preserves_callable_base_when_special_lr_is_none():
    def schedule(step):
        return step * 0.1

    assert resolve_special_lr(schedule, None) is schedule
