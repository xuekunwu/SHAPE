"""Organoid native allocation bounds; no GT, device or model dependencies."""
import math
import os

def check_native_size(shape, diameter, *, augment=False, niter=None):
    cap=int(os.environ.get("SHAPE_ORGANOID_MAX_RESCALED_PIXELS","16000000"))
    if cap<1: raise ValueError("Positive SHAPE_ORGANOID_MAX_RESCALED_PIXELS required")
    scale=30./(30. if diameter is None else float(diameter))
    if not math.isfinite(scale) or scale<=0: raise ValueError("Invalid native diameter")
    pixels=math.ceil(shape[0]*scale)*math.ceil(shape[1]*scale)
    if pixels>cap:
        raise ValueError(f"organoid native_pixel_budget: rescaled pixels {pixels} exceed {cap}")
    return pixels
