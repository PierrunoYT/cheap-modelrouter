"""OpenAI-compatible HTTP server in front of the router.

Lets any OpenAI-compatible client (OpenCode, Cline, Zed, curl, the openai SDK)
use the router as a "model": the client asks for a *mode* (auto / cheap /
balanced / quality) and the router picks the actual model per request.

Start:
  python server.py                       # 127.0.0.1:8787
  python server.py --port 9000 --family deepseek --family qwen

Endpoints:
  POST /v1/chat/completions   (streaming SSE and non-streaming JSON,
                               tools / tool_calls passed through)
  GET  /v1/models             (the four routing modes)
  GET  /health

OpenCode setup (opencode.json):
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "cnrouter": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Chinese Model Router",
        "options": { "baseURL": "http://127.0.0.1:8787/v1" },
        "models": {
          "auto":     { "name": "Router (auto)" },
          "cheap":    { "name": "Router (cheap)" },
          "balanced": { "name": "Router (balanced)" },
          "quality":  { "name": "Router (quality)" }
        }
      }
    },
    "model": "cnrouter/auto"
  }

The server keeps OPENROUTER_API_KEY on this machine; the client never sees it.
Conversations are pinned to their first-routed model via the router's sticky
sessions (keyed by a hash of the first user message), keeping provider prompt
caches warm across agent turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router import EasyChineseModelRouter, StreamResult

MODES = ("auto", "cheap", "balanced", "quality")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_server.py)
# ---------------------------------------------------------------------------

def resolve_mode(model_field: str | None) -> str:
    """Map the request's ``model`` to a routing mode.

    Accepts bare modes ("cheap") and provider-prefixed ids ("cnrouter/cheap").
    Anything unrecognized falls back to "auto" so clients with hardcoded
    model names still work.
    """
    if not model_field:
        return "auto"
    name = str(model_field).rsplit("/", 1)[-1].strip().lower()
    return name if name in MODES else "auto"


def session_id_for(messages: list[dict]) -> str | None:
    """Derive a stable session id from the first user message.

    Agent clients resend the whole conversation each turn, so the first user
    turn identifies the conversation well enough for best-effort stickiness.
    """
    first = EasyChineseModelRouter._last_user_text(list(reversed(messages)))
    if not first:
        return None
    return hashlib.sha1(first[:2000].encode("utf-8", "replace")).hexdigest()[:16]


def completion_payload(result: dict, completion_id: str, created: int) -> dict:
    tool_calls = result.get("tool_calls")
    message: dict = {"role": "assistant", "content": result.get("content") or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish = result.get("finish_reason") or ("tool_calls" if tool_calls else "stop")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": result.get("model", "unknown"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": result.get("usage")
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def chunk_payload(
    completion_id: str,
    created: int,
    model: str,
    *,
    delta: dict | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    payload: dict = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
    }
    if delta is not None or finish_reason is not None:
        payload["choices"] = [
            {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        ]
    if usage is not None:
        payload["usage"] = usage
    return payload


def error_payload(message: str, *, err_type: str = "router_error") -> dict:
    return {"error": {"message": message, "type": err_type, "code": None}}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class RouterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        allow_families: set[str] | None = None,
        sticky: bool = True,
        classifier: str = "keyword",
        classifier_model: str | None = None,
        live_pricing: bool = False,
        log_file: str | None = None,
    ) -> None:
        super().__init__(address, RouterHTTPHandler)
        self.allow_families = allow_families
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.live_pricing = live_pricing
        self.log_file = log_file
        self.sticky_store = (
            os.path.join(tempfile.gettempdir(), "echinese_router_sessions.json")
            if sticky
            else None
        )
        self._routers: dict[str, EasyChineseModelRouter] = {}

    def get_router(self, mode: str) -> EasyChineseModelRouter:
        if mode not in self._routers:
            kwargs: dict = {}
            if self.classifier_model:
                kwargs["classifier_model"] = self.classifier_model
            self._routers[mode] = EasyChineseModelRouter(
                mode=mode,  # type: ignore[arg-type]
                allow_families=self.allow_families,
                sticky_store=self.sticky_store,
                classifier=self.classifier,  # type: ignore[arg-type]
                live_pricing=self.live_pricing,
                log_file=self.log_file,
                **kwargs,
            )
        return self._routers[mode]


class RouterHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RouterServer

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[server] {self.address_string()} {format % args}\n")

    # -- plumbing -----------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length: signal end-of-stream by closing the connection.
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, payload: dict) -> None:
        self.wfile.write(
            b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"
        )
        self.wfile.flush()

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            created = int(time.time())
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": mode, "object": "model", "created": created, "owned_by": "router"}
                        for mode in MODES
                    ],
                },
            )
        elif path in ("", "/health"):
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, error_payload(f"Unknown path: {self.path}", err_type="invalid_request_error"))

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, error_payload(f"Unknown path: {self.path}", err_type="invalid_request_error"))
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self._send_json(400, error_payload("Invalid JSON body", err_type="invalid_request_error"))
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_json(400, error_payload("'messages' must be a non-empty list", err_type="invalid_request_error"))
            return

        mode = resolve_mode(body.get("model"))
        router = self.server.get_router(mode)
        stream = bool(body.get("stream"))

        chat_kwargs = dict(
            messages=messages,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            max_tokens=int(
                body.get("max_tokens") or body.get("max_completion_tokens") or 4096
            ),
            session_id=session_id_for(messages),
            stream=stream,
        )
        if body.get("temperature") is not None:
            chat_kwargs["temperature"] = float(body["temperature"])

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        try:
            result = router.chat(**chat_kwargs)
        except RuntimeError as exc:
            status = 401 if "OPENROUTER_API_KEY" in str(exc) else 502
            self._send_json(status, error_payload(str(exc)))
            return

        if not stream:
            assert isinstance(result, dict)
            self._send_json(200, completion_payload(result, completion_id, created))
            return

        assert isinstance(result, StreamResult)
        model_label = body.get("model") or mode
        self._start_sse()
        try:
            self._sse(chunk_payload(
                completion_id, created, model_label,
                delta={"role": "assistant", "content": ""},
            ))
            for text in result:
                self._sse(chunk_payload(
                    completion_id, created, model_label, delta={"content": text},
                ))
        except Exception as exc:
            # Headers are already sent; report the failure in-band and close.
            self._sse(error_payload(f"{type(exc).__name__}: {exc}"))
            self.wfile.write(b"data: [DONE]\n\n")
            return

        meta = result.meta
        routed_model = meta.get("model", model_label)
        tool_calls = meta.get("tool_calls")

        if tool_calls:
            self._sse(chunk_payload(
                completion_id, created, routed_model,
                delta={
                    "tool_calls": [
                        {"index": i, **tc} for i, tc in enumerate(tool_calls)
                    ]
                },
            ))

        finish = meta.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
        self._sse(chunk_payload(
            completion_id, created, routed_model, finish_reason=finish,
        ))
        if meta.get("usage"):
            self._sse(chunk_payload(
                completion_id, created, routed_model, usage=meta["usage"],
            ))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server for the Chinese model router"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--family",
        action="append",
        help=(
            "Restrict to a family: deepseek, qwen, kimi, glm, minimax, mimo, "
            "inclusionai, tencent, bytedance. Can be repeated."
        ),
    )
    parser.add_argument(
        "--no-sticky",
        action="store_true",
        help="Disable pinning conversations to their first-routed model",
    )
    parser.add_argument(
        "--classifier",
        choices=["keyword", "llm"],
        default="keyword",
        help=(
            "How to detect the task type: 'keyword' (offline regex, default) or "
            "'llm' (ask a cheap OpenRouter model; falls back to keyword on error)."
        ),
    )
    parser.add_argument(
        "--classifier-model",
        help="Model slug used by --classifier llm (default: cheapest in the table)",
    )
    parser.add_argument(
        "--live-pricing",
        action="store_true",
        help="Derive cost scores from OpenRouter's live price list (cached daily)",
    )
    parser.add_argument(
        "--log-file",
        help="Append one JSONL line per request (model, task, latency, real cost)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            "Warning: OPENROUTER_API_KEY is not set; requests will fail with 401.",
            file=sys.stderr,
        )

    server = RouterServer(
        (args.host, args.port),
        allow_families=set(args.family) if args.family else None,
        sticky=not args.no_sticky,
        classifier=args.classifier,
        classifier_model=args.classifier_model,
        live_pricing=args.live_pricing,
        log_file=args.log_file,
    )

    print(f"Router API listening on http://{args.host}:{args.port}/v1")
    print(f"Models (modes): {', '.join(MODES)}")
    print('Point OpenCode at it with "baseURL": '
          f'"http://{args.host}:{args.port}/v1" (see server.py docstring).')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
