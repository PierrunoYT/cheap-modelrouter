"""Tests for server.py, the OpenAI-compatible HTTP layer (no OpenRouter calls).

The integration tests start a real RouterServer on a random localhost port and
inject a stub router, so the full HTTP/SSE path is exercised without network.
"""

import json
import os
import sys
import threading
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router import StreamResult  # noqa: E402
from server import (  # noqa: E402
    RouterServer,
    chunk_payload,
    completion_payload,
    resolve_mode,
    session_id_for,
)


class ResolveModeTests(unittest.TestCase):
    def test_bare_mode(self):
        self.assertEqual(resolve_mode("cheap"), "cheap")

    def test_provider_prefixed(self):
        self.assertEqual(resolve_mode("cnrouter/quality"), "quality")

    def test_unknown_falls_back_to_auto(self):
        self.assertEqual(resolve_mode("gpt-4o"), "auto")
        self.assertEqual(resolve_mode(None), "auto")


class SessionIdTests(unittest.TestCase):
    def test_stable_across_turns(self):
        turn1 = [{"role": "user", "content": "fix my bug"}]
        turn2 = turn1 + [
            {"role": "assistant", "content": "sure"},
            {"role": "user", "content": "thanks, now add tests"},
        ]
        self.assertEqual(session_id_for(turn1), session_id_for(turn2))

    def test_none_without_user_message(self):
        self.assertIsNone(session_id_for([{"role": "system", "content": "s"}]))


class PayloadTests(unittest.TestCase):
    def test_completion_with_tool_calls(self):
        result = {
            "model": "deepseek/deepseek-v4-flash",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "ls", "arguments": "{}"}}],
            "finish_reason": None,
            "usage": None,
        }
        payload = completion_payload(result, "chatcmpl-x", 123)
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_1")
        self.assertIsNone(choice["message"]["content"])

    def test_chunk_shapes(self):
        delta_chunk = chunk_payload("id", 1, "m", delta={"content": "hi"})
        self.assertEqual(delta_chunk["choices"][0]["delta"], {"content": "hi"})
        usage_chunk = chunk_payload("id", 1, "m", usage={"total_tokens": 5})
        self.assertEqual(usage_chunk["choices"], [])
        self.assertEqual(usage_chunk["usage"], {"total_tokens": 5})


class _StubRouter:
    """Stands in for EasyChineseModelRouter; records kwargs, returns canned data."""

    def __init__(self, result=None, stream_chunks=None, stream_meta=None):
        self.result = result
        self.stream_chunks = stream_chunks or []
        self.stream_meta = stream_meta or {}
        self.last_kwargs = None

    def chat(self, prompt="", **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            def gen():
                for chunk in self.stream_chunks:
                    yield chunk
                return self.stream_meta
            return StreamResult(gen())
        return self.result


class HTTPIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = RouterServer(("127.0.0.1", 0), sticky=False)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _post(self, path, payload):
        req = urllib.request.Request(
            self._url(path),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req)

    def _inject(self, mode, stub):
        self.server._routers[mode] = stub
        self.addCleanup(self.server._routers.pop, mode, None)

    def test_models_endpoint(self):
        with urllib.request.urlopen(self._url("/v1/models")) as resp:
            data = json.load(resp)
        self.assertEqual({m["id"] for m in data["data"]},
                         {"auto", "cheap", "balanced", "quality"})

    def test_non_streaming_completion(self):
        stub = _StubRouter(result={
            "family": "deepseek", "model_name": "deepseek_flash",
            "model": "deepseek/deepseek-v4-flash",
            "content": "hello from the router", "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        })
        self._inject("cheap", stub)

        with self._post("/v1/chat/completions", {
            "model": "cnrouter/cheap",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "ls"}}],
        }) as resp:
            data = json.load(resp)

        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(
            data["choices"][0]["message"]["content"], "hello from the router"
        )
        self.assertEqual(data["usage"]["total_tokens"], 8)
        # tools must reach the router untouched
        self.assertEqual(stub.last_kwargs["tools"][0]["function"]["name"], "ls")

    def test_streaming_completion_with_tool_calls(self):
        stub = _StubRouter(
            stream_chunks=["Hel", "lo"],
            stream_meta={
                "model": "z-ai/glm-5.2",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "read", "arguments": "{}"}}],
                "finish_reason": "tool_calls",
                "usage": {"total_tokens": 9},
            },
        )
        self._inject("auto", stub)

        with self._post("/v1/chat/completions", {
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as resp:
            self.assertTrue(resp.headers["Content-Type"].startswith("text/event-stream"))
            raw = resp.read().decode("utf-8")

        events = [
            json.loads(line[len("data: "):])
            for line in raw.strip().splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertTrue(raw.strip().endswith("data: [DONE]"))

        content = "".join(
            e["choices"][0]["delta"].get("content", "")
            for e in events if e.get("choices")
        )
        self.assertEqual(content, "Hello")

        tool_events = [
            e for e in events
            if e.get("choices") and "tool_calls" in e["choices"][0]["delta"]
        ]
        self.assertEqual(len(tool_events), 1)
        tc = tool_events[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["index"], 0)
        self.assertEqual(tc["function"]["name"], "read")

        finishes = [
            e["choices"][0]["finish_reason"]
            for e in events if e.get("choices") and e["choices"][0]["finish_reason"]
        ]
        self.assertEqual(finishes, ["tool_calls"])
        self.assertEqual(events[-1]["usage"]["total_tokens"], 9)

    def test_bad_request_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/v1/chat/completions", {"model": "auto", "messages": []})
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(self._url("/v1/nope"))
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
