from typing import TypeVar


LearningRate = TypeVar("LearningRate")


def resolve_special_lr(
    base_lr: LearningRate,
    special_lr: LearningRate | None,
) -> LearningRate:
    """Use the base learning rate unless a special one is explicitly provided."""
    return base_lr if special_lr is None else special_lr
