"""Shared capture and recovery space estimates."""


def photo_save_budget(raw_frame_bytes: int) -> int:
    return 3 * raw_frame_bytes + 1024**2


# Reserve validation needs headroom even before the first captured bird.
DEFAULT_CAPTURE_BUDGET_BYTES = photo_save_budget(4096 * 2160 * 3)


def recovery_headroom_bytes(budget_bytes: int) -> int:
    return 4 * budget_bytes
