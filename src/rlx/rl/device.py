from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime in train flow
    torch = None  # type: ignore[assignment]


class DeviceResolutionError(Exception):
    """Raised when a requested training device is unsupported or unavailable."""


def resolve_device(requested: str) -> str:
    """Resolve a configured device string to a concrete torch device."""

    if torch is None:
        raise DeviceResolutionError(
            "PyTorch is not installed. Install RL dependencies before running `rlx train`."
        )

    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    try:
        device = torch.device(normalized)
    except (RuntimeError, TypeError) as exc:
        raise DeviceResolutionError(f"Unsupported device: {requested}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceResolutionError("CUDA was requested but is not available on this machine.")
    if device.type == "mps" and not _mps_available():
        raise DeviceResolutionError("MPS was requested but is not available on this machine.")

    return str(device)


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
