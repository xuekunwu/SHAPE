"""Reasoning profiles for stage execution, independent of dataset sampling."""

def reasoning_protocol(profile: str = "all-high") -> dict:
    # All four reasoning stages; independent of CPJUMP1's medium image-stage defaults.
    if profile not in ("all-high", "image-medium"):
        raise ValueError(f"unknown reasoning profile: {profile}")
    protocol = {"arms": ["shape", "codex"], "table": {
        stage:{"effort":"high", "codex_effort":"high", "max_tokens":16000, "timeout_seconds":900}
        for stage in ("segment", "crop", "embed", "batch")}}
    if profile == "image-medium":
        for stage in ("segment", "crop"):
            protocol["table"][stage]["effort"] = "medium"
    return protocol
