"""Model API transport: one OpenAI-compatible chat call, stdlib only.

Configuration (env var names, path, timeout) lives in
``conf/transport.yaml``; secrets and values live only in environment
variables. Unit tests inject a fake opener and make no network calls.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

_CONFIG_PATH = Path(__file__).parent / "conf" / "transport.yaml"

logger = logging.getLogger("zydah.transport")

# opener(url, data, headers, timeout) -> parsed JSON response dict
Opener = Callable[[str, bytes, Dict[str, str], float], Mapping]
# stream_opener(url, data, headers, timeout) -> iterable of raw SSE lines
StreamOpener = Callable[[str, bytes, Dict[str, str], float], Iterable[bytes]]


def load_transport_config() -> DictConfig:
    return OmegaConf.load(_CONFIG_PATH)


_LAST_STATS: Dict[str, object] = {}
_CUMULATIVE: Dict[str, object] = {
    "calls": 0,
    "elapsed_s": 0.0,
    "tokens_in": None,
    "tokens_out": None,
}


def cumulative_stats() -> Dict[str, object]:
    """Running totals over all completed calls in this process (copy).

    Aggregators diff two snapshots to price a span that may contain
    several calls (e.g. a 3-judge panel elicitation).
    """
    return dict(_CUMULATIVE)


def last_call_stats() -> Dict[str, object]:
    """Stats for the most recent completed model call (copy).

    Keys: ``seq`` (monotonic call counter), ``model``, ``stream``,
    ``elapsed_s``, ``tokens_in``, ``tokens_out`` (None when the
    endpoint did not report usage). Empty before any call completes.
    Callers that aggregate (e.g. the verifier) diff ``seq`` around a
    call instead of trusting wall-clock bookkeeping of their own.
    """
    return dict(_LAST_STATS)


def _record_stats(
    model: str,
    started: float,
    usage: Optional[Mapping],
    *,
    stream: bool,
) -> None:
    tokens_in = tokens_out = None
    if isinstance(usage, Mapping):
        if isinstance(usage.get("prompt_tokens"), int):
            tokens_in = usage["prompt_tokens"]
        if isinstance(usage.get("completion_tokens"), int):
            tokens_out = usage["completion_tokens"]
    elapsed = time.perf_counter() - started
    _LAST_STATS.update(
        {
            "seq": int(_LAST_STATS.get("seq", 0)) + 1,
            "model": model,
            "stream": stream,
            "elapsed_s": elapsed,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
    )
    _CUMULATIVE["calls"] = int(_CUMULATIVE["calls"]) + 1  # type: ignore[arg-type]
    _CUMULATIVE["elapsed_s"] = float(_CUMULATIVE["elapsed_s"]) + elapsed  # type: ignore[arg-type]
    for key, value in (("tokens_in", tokens_in), ("tokens_out", tokens_out)):
        if isinstance(value, int):
            current = _CUMULATIVE[key]
            _CUMULATIVE[key] = (current if isinstance(current, int) else 0) + value
    logger.info(
        "llm call model=%s stream=%s elapsed_s=%.3f tokens_in=%s tokens_out=%s",
        model,
        stream,
        elapsed,
        tokens_in,
        tokens_out,
    )


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
    reasoning_effort: Optional[str] = None,
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
    effort = reasoning_effort or OmegaConf.select(cfg, "reasoning_effort")
    if effort:
        payload["reasoning_effort"] = str(effort)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    open_fn = opener or _default_opener
    timeout = float(OmegaConf.select(cfg, "timeout_seconds"))
    retries = int(OmegaConf.select(cfg, "retries"))
    backoff = float(OmegaConf.select(cfg, "retry_backoff_seconds"))
    data = json.dumps(payload).encode("utf-8")
    logger.info("chat_complete model=%s prompt:\n%s", model, prompt)
    started = time.perf_counter()
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
    text = content if isinstance(content, str) else str(content)
    usage = response.get("usage") if isinstance(response, Mapping) else None
    _record_stats(model, started, usage, stream=False)
    logger.info("chat_complete response:\n%s", text)
    return text


def _default_stream_opener(
    url: str, data: bytes, headers: Dict[str, str], timeout: float
) -> Iterable[bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(request, timeout=timeout)


def stream_chat_complete(
    prompt: str,
    model_id: Optional[str] = None,
    *,
    config: Optional[DictConfig] = None,
    opener: Optional[StreamOpener] = None,
) -> Iterator[str]:
    """Stream one chat call; yield content deltas as they arrive (SSE).

    Same env/config resolution as ``chat_complete``, with
    ``stream: true``. No retries: a stream that has already yielded
    tokens cannot be replayed honestly.
    """
    cfg = config or load_transport_config()
    base_url = _env_or_raise(cfg, "base_url_env")
    api_key = _env_or_raise(cfg, "api_key_env")
    env_model = os.environ.get(str(OmegaConf.select(cfg, "model_env")))
    model = env_model or model_id or _env_or_raise(cfg, "model_env")
    url = base_url.rstrip("/") + str(OmegaConf.select(cfg, "chat_path"))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if OmegaConf.select(cfg, "stream_usage"):
        payload["stream_options"] = {"include_usage": True}
    effort = OmegaConf.select(cfg, "reasoning_effort")
    if effort:
        payload["reasoning_effort"] = str(effort)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    open_fn = opener or _default_stream_opener
    timeout = float(OmegaConf.select(cfg, "timeout_seconds"))
    data = json.dumps(payload).encode("utf-8")
    logger.info("stream_chat_complete model=%s prompt:\n%s", model, prompt)
    started = time.perf_counter()
    try:
        response = open_fn(url, data, headers, timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"model API HTTP {exc.code} for model {model!r}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"model API unreachable: {reason}") from exc
    usage_out: Dict[str, object] = {}
    try:
        chunks: List[str] = []
        for delta in _sse_content(response, usage_out):
            chunks.append(delta)
            yield delta
        _record_stats(
            model,
            started,
            usage_out.get("usage"),  # type: ignore[arg-type]
            stream=True,
        )
        logger.info("stream_chat_complete response (%d chunks):\n%s", len(chunks), "".join(chunks))
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"model API stream interrupted: {reason}") from exc


def _raw_lines(response) -> Iterator[bytes]:
    """Yield lines as bytes land, not as a buffered reader fills.

    ``read1`` performs at most one raw socket read per call, so a
    short SSE event is delivered immediately instead of waiting for
    an 8 KiB buffer to fill. Falls back to line iteration for
    file-like test doubles without ``read1``.
    """
    read1 = getattr(response, "read1", None)
    if not callable(read1):
        yield from response
        return
    buffer = b""
    while True:
        data = read1(4096)
        if not data:
            break
        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line
    if buffer:
        yield buffer


def _sse_content(response, usage_out: Optional[Dict] = None) -> Iterator[str]:
    """Parse SSE deltas; tolerate a plain JSON (non-streamed) reply.

    When ``usage_out`` is given, the last ``usage`` object seen (the
    final chunk under ``stream_options.include_usage``, or the body of
    a plain JSON reply) is stored under ``usage_out["usage"]``.
    """
    pending_json: List[bytes] = []
    for raw in _raw_lines(response):
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        stripped = line.strip()
        if pending_json:
            pending_json.append(raw if isinstance(raw, bytes) else line.encode())
            continue
        if not stripped.startswith("data:"):
            if stripped.startswith("{"):
                # Endpoint ignored stream:true and sent one JSON body.
                pending_json.append(
                    raw if isinstance(raw, bytes) else line.encode()
                )
            continue
        chunk = stripped[5:].strip()
        if chunk == "[DONE]":
            return
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        if usage_out is not None and isinstance(obj.get("usage"), Mapping):
            usage_out["usage"] = obj["usage"]
        try:
            delta = obj["choices"][0].get("delta", {}).get("content")
        except (KeyError, IndexError, TypeError):
            continue
        if delta:
            yield str(delta)
    if pending_json:
        try:
            obj = json.loads(b"".join(pending_json).decode("utf-8", "replace"))
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            return
        if usage_out is not None and isinstance(obj.get("usage"), Mapping):
            usage_out["usage"] = obj["usage"]
        if content:
            yield str(content)
