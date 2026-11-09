"""Pure percentage metrics; no fitting, file IO or private-label access."""
import numpy as np


def metrics(target, predicted) -> dict:
    y, p = np.asarray(target, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("invalid targets or predictions")
    if np.any((y < 0) | (y > 100)) or np.any((p < 0) | (p > 100)):
        raise ValueError("predictions and targets must be percentages in [0, 100]")
    error = p-y
    denominator = float(np.sum((y-y.mean())**2))
    return {"wells": len(y), "mae_percentage_points": float(np.mean(np.abs(error))),
            "rmse_percentage_points": float(np.sqrt(np.mean(error**2))),
            "bias_percentage_points": float(np.mean(error)),
            "r2": float(1-np.sum(error**2)/denominator) if len(y)>1 and denominator>0 else None}
