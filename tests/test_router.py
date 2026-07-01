"""Unit tests for router.py (no network calls)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router import (  # noqa: E402
    EasyChineseModelRouter,
    ModelProfile,
    StreamResult,
)


class EstimateTokensTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(EasyChineseModelRouter.estimate_tokens(""), 1)

    def test_english_ratio(self):
        # ~4 chars per token for Latin text.
        text = "a" * 40
        self.assertEqual(EasyChineseModelRouter.estimate_tokens(text), 10)

    def test_cjk_ratio(self):
        # 2 tokens per CJK char (conservative).
        text = "中" * 10
        self.assertEqual(EasyChineseModelRouter.estimate_tokens(text), 20)

    def test_mixed(self):
        # 5 CJK (10 tokens) + 8 latin (2 tokens) = 12
        self.assertEqual(EasyChineseModelRouter.estimate_tokens("你好世界吗abcdefgh"), 12)

    def test_cjk_more_than_english(self):
        cjk = EasyChineseModelRouter.estimate_tokens("数据库连接超时错误")
        latin = EasyChineseModelRouter.estimate_tokens("database timeout")
        self.assertGreater(cjk, latin)


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.router = EasyChineseModelRouter()

    def test_story_is_creative_not_coding(self):
        # The historical false positive: "bug" inside a creative prompt.
        self.assertEqual(
            self.router.classify("write a short story about bugs"),
            "creative",
        )

    def test_code_fence_is_coding(self):
        self.assertEqual(self.router.classify("fix this:\n```\ndef f(): pass\n```"), "coding")

    def test_translation(self):
        self.assertEqual(self.router.classify("translate this paragraph"), "translation")

    def test_reasoning(self):
        self.assertEqual(self.router.classify("analyze the tradeoffs here"), "reasoning")

    def test_simple_default(self):
        self.assertEqual(self.router.classify("hello there"), "simple")

    def test_long_context(self):
        # 4 non-space chars/word -> ~80k tokens, above the 70k threshold.
        big = "word " * 80_000
        self.assertEqual(self.router.classify(big), "long_context")


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.router = EasyChineseModelRouter()

    def test_route_returns_models(self):
        route = self.router.route("hello")
        self.assertTrue(route)
        self.assertIsInstance(route[0], ModelProfile)

    def test_family_filter(self):
        router = EasyChineseModelRouter(allow_families={"deepseek"})
        route = router.route("hello")
        self.assertTrue(route)
        self.assertTrue(all(m.family == "deepseek" for m in route))

    def test_long_context_excludes_small_windows(self):
        # ~800k tokens leaves only models with ~1M context windows.
        big_prompt = "x " * 3_200_000
        route = self.router.route(big_prompt)
        self.assertTrue(route)
        for m in route:
            self.assertGreaterEqual(m.max_context_tokens, 1_000_000)

    def test_unknown_family_yields_empty(self):
        router = EasyChineseModelRouter(allow_families={"nonexistent"})
        self.assertEqual(router.route("hello"), [])


class ToolSupportTests(unittest.TestCase):
    def _profile(self, name, supports_tools):
        return ModelProfile(
            name=name, family="t", model=f"t/{name}",
            cost_score=1.0, quality_score=5.0,
            supports_tools=supports_tools,
        )

    def test_require_tools_filters_models(self):
        router = EasyChineseModelRouter(
            models=[self._profile("with_tools", True), self._profile("no_tools", False)]
        )
        names = [m.name for m in router.route("hello", require_tools=True)]
        self.assertEqual(names, ["with_tools"])

    def test_no_filter_without_require_tools(self):
        router = EasyChineseModelRouter(
            models=[self._profile("with_tools", True), self._profile("no_tools", False)]
        )
        self.assertEqual(len(router.route("hello")), 2)

    def test_merge_tool_call_delta(self):
        from types import SimpleNamespace as NS

        acc = {}
        # id + name arrive first, arguments stream in fragments.
        EasyChineseModelRouter._merge_tool_call_delta(
            acc, NS(index=0, id="call_1", function=NS(name="read_file", arguments='{"pa'))
        )
        EasyChineseModelRouter._merge_tool_call_delta(
            acc, NS(index=0, id=None, function=NS(name=None, arguments='th": "a.py"}'))
        )
        EasyChineseModelRouter._merge_tool_call_delta(
            acc, NS(index=1, id="call_2", function=NS(name="ls", arguments="{}"))
        )

        self.assertEqual(acc[0]["id"], "call_1")
        self.assertEqual(acc[0]["function"]["name"], "read_file")
        self.assertEqual(acc[0]["function"]["arguments"], '{"path": "a.py"}')
        self.assertEqual(acc[1]["function"]["name"], "ls")

    def test_last_user_text_string_and_parts(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": [
                {"type": "text", "text": "fix"},
                {"type": "text", "text": "this bug"},
            ]},
        ]
        self.assertEqual(
            EasyChineseModelRouter._last_user_text(messages), "fix this bug"
        )


class LlmClassifierTests(unittest.TestCase):
    """LLM classifier dispatch, parsing and fallback -- no network calls."""

    def _router(self, reply=None, error=None):
        router = EasyChineseModelRouter(classifier="llm")
        calls = []

        def fake_classify_llm(prompt):
            calls.append(prompt)
            if error:
                raise error
            return reply

        router._classify_llm = fake_classify_llm
        router._llm_calls = calls
        return router

    def test_uses_llm_label(self):
        router = self._router(reply="creative")
        self.assertEqual(router.classify("name my new coffee shop"), "creative")

    def test_falls_back_to_keyword_on_error(self):
        router = self._router(error=RuntimeError("offline"))
        self.assertEqual(router.classify("translate this paragraph"), "translation")

    def test_long_context_never_calls_llm(self):
        router = self._router(reply="simple")
        big = "word " * 80_000
        self.assertEqual(router.classify(big), "long_context")
        self.assertEqual(router._llm_calls, [])

    def test_result_is_cached_per_prompt(self):
        router = self._router(reply="coding")
        router.classify("fix my thing")
        router.classify("fix my thing")
        self.assertEqual(len(router._llm_calls), 1)

    def test_keyword_default_never_calls_llm(self):
        router = EasyChineseModelRouter()  # classifier="keyword"
        router._classify_llm = lambda prompt: self.fail("LLM classifier called")
        self.assertEqual(router.classify("hello there"), "simple")

    def test_parse_tolerates_chatty_reply(self):
        # Exercise the real reply-parsing logic with a fake API client.
        from types import SimpleNamespace as NS

        router = EasyChineseModelRouter(classifier="llm")
        response = NS(choices=[NS(message=NS(content="Category: Coding."))])
        client = NS(chat=NS(completions=NS(create=lambda **kw: response)))
        router._build_client = lambda api_key: client

        os.environ["OPENROUTER_API_KEY"] = "sk-test-dummy"
        self.assertEqual(router._classify_llm("whatever"), "coding")

    def test_parse_rejects_junk_reply(self):
        from types import SimpleNamespace as NS

        router = EasyChineseModelRouter(classifier="llm")
        response = NS(choices=[NS(message=NS(content="I cannot classify that"))])
        client = NS(chat=NS(completions=NS(create=lambda **kw: response)))
        router._build_client = lambda api_key: client

        os.environ["OPENROUTER_API_KEY"] = "sk-test-dummy"
        with self.assertRaises(ValueError):
            router._classify_llm("whatever")


class ErrorClassificationTests(unittest.TestCase):
    class _Exc(Exception):
        def __init__(self, status_code=None):
            super().__init__("boom")
            self.status_code = status_code

    def test_auth_error_by_status(self):
        self.assertTrue(EasyChineseModelRouter._is_auth_error(self._Exc(401)))
        self.assertTrue(EasyChineseModelRouter._is_auth_error(self._Exc(403)))

    def test_retryable_status(self):
        for code in (429, 500, 502, 503, 504):
            self.assertTrue(EasyChineseModelRouter._is_retryable(self._Exc(code)))

    def test_bad_request_not_retryable(self):
        self.assertTrue(EasyChineseModelRouter._is_bad_request(self._Exc(400)))
        self.assertFalse(EasyChineseModelRouter._is_retryable(self._Exc(400)))


class ReasoningBodyTests(unittest.TestCase):
    def _model(self, reasoning):
        return ModelProfile(
            name="t", family="t", model="t/t",
            cost_score=1.0, quality_score=1.0, reasoning=reasoning,
        )

    def test_reasoning_enabled_for_coding(self):
        body = EasyChineseModelRouter._reasoning_body(self._model(True), "coding")
        self.assertEqual(body, {"reasoning": {"enabled": True}})

    def test_no_reasoning_for_simple(self):
        self.assertEqual(EasyChineseModelRouter._reasoning_body(self._model(True), "simple"), {})

    def test_no_reasoning_when_unsupported(self):
        self.assertEqual(EasyChineseModelRouter._reasoning_body(self._model(False), "coding"), {})


class StreamResultTests(unittest.TestCase):
    def test_yields_and_captures_meta(self):
        def gen():
            yield "Hello "
            yield "world"
            return {"family": "deepseek", "model": "x", "usage": None}

        sr = StreamResult(gen())
        self.assertEqual("".join(sr), "Hello world")
        self.assertEqual(sr.meta["family"], "deepseek")

    def test_empty_meta_when_no_return(self):
        def gen():
            yield "hi"

        sr = StreamResult(gen())
        list(sr)
        self.assertEqual(sr.meta, {})


class DryRunTests(unittest.TestCase):
    def test_dry_run_no_network(self):
        os.environ["OPENROUTER_API_KEY"] = "sk-test-dummy"
        router = EasyChineseModelRouter()
        result = router.chat("translate hello", dry_run=True)
        self.assertEqual(result["task"], "translation")
        self.assertIn("fallback_order", result)


if __name__ == "__main__":
    unittest.main()
