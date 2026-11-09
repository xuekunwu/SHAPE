"""Verified provider routing, including an opt-in credential-free SSH loopback proxy."""
import json
import os
import re
import urllib.parse
import urllib.request

STANFORD = "https://llm-gateway.example.edu/v1"
DIRECT = (STANFORD, "https://api.openai.com/v1")
PLACEHOLDER = "local-proxy-no-upstream-credential"


def validate(base_url):
    base_url = base_url.rstrip("/")
    if base_url in DIRECT:
        return {"base_url": base_url, "upstream": base_url, "transport": "direct"}
    url = urllib.parse.urlsplit(base_url)
    if (os.environ.get("SHAPE_LOCAL_GATEWAY_PROXY") != "ssh-loopback-v1" or url.scheme != "http" or
            url.hostname != "127.0.0.1" or not url.port or url.path != "/v1" or
            url.username or url.password or url.query or url.fragment or
            os.environ.get("LLM_API_KEY") != PLACEHOLDER):
        raise ValueError("unsupported gateway route; an explicit verified SSH loopback proxy is required")
    expected = os.environ.get("SHAPE_LOCAL_GATEWAY_PROXY_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("local gateway proxy code hash is missing")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.removesuffix("/v1") + "/__shape_proxy_identity", timeout=10) as response:
        identity = json.load(response)
    if (identity.get("upstream") != STANFORD or identity.get("proxy_sha256") != expected or
            identity.get("credentials") != "local-only"):
        raise ValueError("local gateway proxy identity mismatch")
    return {"base_url": base_url, "upstream": STANFORD, "transport": "ssh-loopback-v1",
            "proxy_sha256": expected, "credential_location": "local workstation; no upstream credential on VM"}
