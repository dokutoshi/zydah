"""Model API transport: one OpenAI-compatible chat call, stdlib only.

Configuration (env var names, path, timeout) lives in
``conf/transport.yaml``; secrets and values live only in environment
variables. Unit tests inject a fake opener and make no network calls.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

_CONFIG_PATH = Path(__file__).parent / "conf" / "transport.yaml"

# opener(url, data, headers, timeout) -> parsed JSON response dict
Opener = Callable[[str, bytes, Dict[str, str], float], Mapping]


def load_transport_config() -> DictConfig:
    return OmegaConf.load(_CONFIG_PATH)


def _default_opener(url: str, data: bytes, headers: Dict[str, str], timeout: float) -> Mapping:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _env_or_raise(cfg: DictConfig, key: str) -> str:
    env_name = str(OmegaConf.select(cfg, key))
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"missing required environment variable {env_name}")
    return value


def chat_complete(
    prompt: str,
    model_id: Optional[str] = None,
    *,
    config: Optional[DictConfig] = None,
    opener: Optional[Opener] = None,
) -> str:
    """Send one user prompt; return the assistant's text content.

    Model resolution: the env var named by ``model_env`` in
    transport.yaml wins when set (the endpoint's configured deployment);
    otherwise the caller's ``model_id`` is sent. Raises RuntimeError on
    missing env or HTTP errors; never includes the API key in messages.
    """
    cfg = config or load_transport_config()
    base_url = _env_or_raise(cfg, "base_url_env")
    api_key = _env_or_raise(cfg, "api_key_env")
    env_model = os.environ.get(str(OmegaConf.select(cfg, "model_env")))
    model = env_model or model_id or _env_or_raise(cfg, "model_env")
    url = base_url.rstrip("/") + str(OmegaConf.select(cfg, "chat_path"))
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    open_fn = opener or _default_opener
    timeout = float(OmegaConf.select(cfg, "timeout_seconds"))
    retries = int(OmegaConf.select(cfg, "retries"))
    backoff = float(OmegaConf.select(cfg, "retry_backoff_seconds"))
    data = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        try:
            response = open_fn(url, data, headers, timeout)
            break
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"model API HTTP {exc.code} for model {model!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                reason = getattr(exc, "reason", exc)
                raise RuntimeError(f"model API unreachable: {reason}") from exc
            attempt += 1
            time.sleep(backoff * attempt)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected model API response shape: {sorted(response)!r}") from exc
    return content if isinstance(content, str) else str(content)
