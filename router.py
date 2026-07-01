"""
Easy Chinese Model Router

One-key setup router for MiniMax, MiMo, Qwen, Kimi, GLM and DeepSeek via OpenRouter.

Quick start:
  1) Create .env with OPENROUTER_API_KEY=sk-or-...
  2) pip install openai python-dotenv
  3) python router.py --prompt "fix this Python bug" --dry-run
  4) python router.py --prompt "write a FastAPI endpoint"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

TaskKind = Literal[
    "simple",
    "coding",
    "reasoning",
    "long_context",
    "translation",
    "creative",
]

Mode = Literal["auto", "cheap", "balanced", "quality"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SYSTEM = "You are a helpful, accurate assistant. Be concise unless detail is needed."


@dataclass(frozen=True)
class ModelProfile:
    name: str
    family: str
    model: str
    cost_score: float          # lower = cheaper
    quality_score: float       # higher = better
    strengths: set[TaskKind] = field(default_factory=set)
    max_context_tokens: int = 128_000
    notes: str = ""
    supports_tools: bool = True
    reasoning: bool = False


# Model slugs may change on OpenRouter, so keep these editable.
#   Use: python router.py --list-models     to print this table
#   Use: python router.py --validate-models  to check slugs against the live catalog
#
# NOTE: cost_score and quality_score are SUBJECTIVE HEURISTICS for relative
# ordering only -- they are not derived from OpenRouter pricing or published
# benchmarks. Tune them to taste; they affect routing order, not billing.
MODELS: list[ModelProfile] = [
    # DeepSeek
    ModelProfile(
        name="deepseek_flash",
        family="deepseek",
        model="deepseek/deepseek-v4-flash",
        cost_score=1.0,
        quality_score=7.6,
        strengths={"simple", "coding", "long_context", "translation"},
        max_context_tokens=1_000_000,
        notes="Cheap default. Good for simple/coding/long-context first pass.",
        reasoning=True,
    ),
    ModelProfile(
        name="deepseek_pro",
        family="deepseek",
        model="deepseek/deepseek-v4-pro",
        cost_score=2.2,
        quality_score=8.5,
        strengths={"reasoning", "coding", "long_context"},
        max_context_tokens=1_000_000,
        notes="Stronger DeepSeek option for reasoning and larger coding tasks.",
        reasoning=True,
    ),

    # Qwen
    ModelProfile(
        name="qwen_flash",
        family="qwen",
        model="qwen/qwen3.6-flash",
        cost_score=1.4,
        quality_score=7.7,
        strengths={"simple", "translation", "creative", "long_context"},
        max_context_tokens=1_000_000,
        notes="Fast and low-cost Qwen model.",
    ),
    ModelProfile(
        name="qwen_max",
        family="qwen",
        model="qwen/qwen3.7-max",
        cost_score=3.4,
        quality_score=8.8,
        strengths={"coding", "reasoning", "creative"},
        max_context_tokens=262_000,
        notes="Flagship Qwen (3.7) for coding and reasoning.",
        reasoning=True,
    ),

    # Kimi / Moonshot
    ModelProfile(
        name="kimi_code",
        family="kimi",
        model="moonshotai/kimi-k2.7-code",
        cost_score=2.7,
        quality_score=8.7,
        strengths={"coding", "reasoning", "long_context"},
        max_context_tokens=262_000,
        notes="Strong coding/agent model from Moonshot Kimi family.",
        reasoning=True,
    ),

    # GLM / Z.ai
    ModelProfile(
        name="glm_5_2",
        family="glm",
        model="z-ai/glm-5.2",
        cost_score=2.5,
        quality_score=8.9,
        strengths={"reasoning", "coding", "long_context"},
        max_context_tokens=1_000_000,
        notes="High-quality GLM route for reasoning, coding and long tasks.",
        reasoning=True,
    ),

    # MiniMax
    ModelProfile(
        name="minimax_m3",
        family="minimax",
        model="minimax/minimax-m3",
        cost_score=2.1,
        quality_score=8.4,
        strengths={"coding", "reasoning", "long_context", "creative"},
        max_context_tokens=524_000,
        notes="More capable MiniMax route with long-context support.",
    ),

    # Xiaomi MiMo
    ModelProfile(
        name="mimo_2_5",
        family="mimo",
        model="xiaomi/mimo-v2.5",
        cost_score=1.2,
        quality_score=8.2,
        strengths={"coding", "reasoning", "long_context", "simple"},
        max_context_tokens=1_000_000,
        notes="Low-cost MiMo route, good for agentic and long-context work.",
        reasoning=True,
    ),
    ModelProfile(
        name="mimo_2_5_pro",
        family="mimo",
        model="xiaomi/mimo-v2.5-pro",
        cost_score=2.4,
        quality_score=9.0,
        strengths={"coding", "reasoning", "long_context"},
        max_context_tokens=1_000_000,
        notes="Flagship MiMo route for harder coding/agent tasks.",
        reasoning=True,
    ),
]


class StreamResult:
    """Iterator over streamed content chunks.

    Yields content strings. After iteration completes, ``meta`` holds the
    final metadata dict (family, model, finish_reason, usage, full content).
    """

    def __init__(self, chunks: "Iterator[str]") -> None:
        self._chunks = chunks
        self.meta: dict = {}

    def __iter__(self) -> "StreamResult":
        return self

    def __next__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration as stop:
            self.meta = stop.value or {}
            raise


class EasyChineseModelRouter:
    def __init__(
        self,
        models: Iterable[ModelProfile] = MODELS,
        *,
        mode: Mode = "auto",
        max_retries_per_model: int = 3,
        allow_families: set[str] | None = None,
        sticky_ttl: float = 300.0,
        sticky_store: str | None = None,
    ) -> None:
        self.models = list(models)
        self.mode = mode
        self.max_retries_per_model = max_retries_per_model
        self.allow_families = {f.lower() for f in allow_families} if allow_families else None

        # Session stickiness: once a session routes to a model, reuse it for
        # follow-up turns so the provider prompt cache stays warm and the
        # assistant's voice/quality stays consistent. Mirrors OpenRouter's
        # ~5-minute session pinning. Maps session_id -> (model_name, expires_at).
        #
        # In-memory by default (great for a long-lived router object). The CLI
        # runs a fresh process per turn, so it passes ``sticky_store`` to persist
        # the mapping to a small JSON file keyed by session_id.
        self.sticky_ttl = sticky_ttl
        self.sticky_store = sticky_store
        self._sticky_choices: dict[str, tuple[str, float]] = self._load_sticky()

    def _load_sticky(self) -> dict[str, tuple[str, float]]:
        if not self.sticky_store or not os.path.exists(self.sticky_store):
            return {}
        try:
            with open(self.sticky_store, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            now = time.time()
            return {
                sid: (entry[0], float(entry[1]))
                for sid, entry in raw.items()
                if float(entry[1]) > now  # drop already-expired sessions on load
            }
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return {}  # corrupt/old store: start clean rather than crash

    def _save_sticky(self) -> None:
        if not self.sticky_store:
            return
        try:
            with open(self.sticky_store, "w", encoding="utf-8") as fh:
                json.dump(self._sticky_choices, fh)
        except OSError:
            pass  # persistence is best-effort; never fail a chat over it

    def _apply_sticky(
        self, session_id: str | None, route: list[ModelProfile]
    ) -> list[ModelProfile]:
        """Move a session's remembered model to the front of the route.

        The pinned model is only honored if it is still a valid candidate for
        this request (i.e. still in ``route`` after context/family filtering),
        so a grown context window or a family restriction transparently
        re-routes instead of forcing an unusable model.
        """
        if not session_id:
            return route

        entry = self._sticky_choices.get(session_id)
        if not entry:
            return route

        name, expires_at = entry
        if time.time() > expires_at:
            self._sticky_choices.pop(session_id, None)
            self._save_sticky()
            return route

        pinned = next((m for m in route if m.name == name), None)
        if pinned is None:
            return route

        return [pinned, *(m for m in route if m.name != name)]

    def _remember_sticky(self, session_id: str | None, model_name: str) -> None:
        if session_id:
            self._sticky_choices[session_id] = (model_name, time.time() + self.sticky_ttl)
            self._save_sticky()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Heuristic token estimate, CJK-aware.
        # Sources: OpenAI tokenizer docs (~4 chars/token for English),
        # presenc.ai 2026 research (~1.4-1.8 tokens/CJK char on GPT-style tokenizers),
        # tianpan.co guidance (use 2 tokens/CJK char for safe budgeting).
        # We use the conservative 2.0 multiplier because this estimate guards
        # context-window overflow, where undercounting is worse than overcounting.
        if not text:
            return 1

        cjk_chars = 0
        other_chars = 0

        for ch in text:
            code = ord(ch)
            if (
                0x4E00 <= code <= 0x9FFF        # CJK Unified Ideographs
                or 0x3400 <= code <= 0x4DBF     # CJK Extension A
                or 0x20000 <= code <= 0x2A6DF   # CJK Extension B
                or 0x3040 <= code <= 0x30FF     # Hiragana + Katakana
                or 0xAC00 <= code <= 0xD7AF     # Hangul Syllables
                or 0xFF00 <= code <= 0xFFEF     # Fullwidth forms
            ):
                cjk_chars += 1
            elif not ch.isspace():
                other_chars += 1

        return max(1, cjk_chars * 2 + other_chars // 4)

    def classify(self, prompt: str, messages: list[dict] | None = None) -> TaskKind:
        text = prompt.lower()
        token_estimate = self.estimate_tokens(
            prompt if messages is None else json.dumps(messages, ensure_ascii=False)
        )

        if token_estimate > 70_000:
            return "long_context"

        # Each category carries both English (\b word-boundary) patterns and
        # Chinese/CJK patterns. CJK has no word boundaries, so those are bare
        # substring alternations -- important for a Chinese-model router, where
        # prompts like "用 Python 写一个排序函数" must classify correctly.

        # Strong code signals (fenced code, language syntax) win outright.
        if any(re.search(pattern, text) for pattern in [
            r"```|\bdef \b|\bclass \b|\bfunction \b|\bconst \b|\blet \b|\bimport \b",
            r"\b(stack trace|traceback|stacktrace|segfault|compile error|syntax error)\b",
            r"(堆栈|回溯|报错|编译错误|段错误|异常堆栈)",
        ]):
            return "coding"

        # Explicit creative-writing intent is checked before generic coding
        # keywords so "write a story about bugs" is not misread as coding.
        if any(re.search(pattern, text) for pattern in [
            r"\bwrite (me )?a (short )?(story|poem|song|script|essay|tagline|slogan|headline)\b",
            r"\bwrite a prompt\b",
            r"\bmake it sound\b",
            r"\b(brainstorm|come up with) (some )?(ideas|names|titles)\b",
            r"写.{0,15}(故事|诗|诗歌|歌词|剧本|散文|文案|标语|口号|广告语|对联)",
            r"(起.{0,2}名|取.{0,2}名|想.{0,4}名字|起标题|头条标题)",
        ]):
            return "creative"

        # Translation / language editing.
        if any(re.search(pattern, text) for pattern in [
            r"\btranslate\b",
            r"\bfix (the )?grammar\b",
            r"\brewrite (this )?in\b",
            r"\binto (chinese|english|japanese|korean|spanish|french|german)\b",
            r"(翻译|译成|改写成|润色|语法错误)",
        ]):
            return "translation"

        # Generic coding keywords (weaker than the strong signals above).
        if any(re.search(pattern, text) for pattern in [
            r"\b(code|coding|debug|refactor|bug fix|stack overflow|repo|git|pull request|commit|diff|patch)\b",
            r"\b(python|typescript|javascript|node|react|fastapi|powershell|bash|sql|docker|kubernetes)\b",
            r"(代码|编程|调试|重构|函数|算法|脚本|程序(?!员)|接口|报错|修复.{0,3}bug)",
            r"写.{0,6}(函数|程序|脚本|类|接口|代码|算法)",
        ]):
            return "coding"

        if any(re.search(pattern, text) for pattern in [
            r"\b(reason|think|analyze|architecture|compare|tradeoff|prove|math|optimize|evaluate)\b",
            r"\b(why|which is better|best approach|design|plan)\b",
            r"(分析|推理|为什么|比较|权衡|证明|数学|优化|评估|架构|设计|规划|哪个更好|最佳方案|怎么实现|如何实现)",
        ]):
            return "reasoning"

        return "simple"

    def route(
        self,
        prompt: str,
        messages: list[dict] | None = None,
        *,
        require_tools: bool = False,
    ) -> list[ModelProfile]:
        task = self.classify(prompt, messages)
        estimated_tokens = self.estimate_tokens(
            prompt if messages is None else json.dumps(messages, ensure_ascii=False)
        )

        candidates = [
            model
            for model in self.models
            if estimated_tokens <= int(model.max_context_tokens * 0.85)
            and (self.allow_families is None or model.family.lower() in self.allow_families)
            and (not require_tools or model.supports_tools)
        ]

        if not candidates:
            return []

        def score(model: ModelProfile) -> tuple[float, float, str]:
            task_bonus = -1.0 if task in model.strengths else 1.5

            if self.mode == "cheap":
                return (model.cost_score + task_bonus, -model.quality_score, model.name)

            if self.mode == "quality":
                return (-model.quality_score + task_bonus, model.cost_score, model.name)

            if self.mode == "balanced":
                return (
                    model.cost_score * 0.55 - model.quality_score * 0.45 + task_bonus,
                    model.cost_score,
                    model.name,
                )

            # auto mode:
            # Cheap for simple tasks, stronger for coding/reasoning/long context.
            if task in {"simple", "translation", "creative"}:
                return (model.cost_score + task_bonus, -model.quality_score, model.name)

            return (
                model.cost_score * 0.4 - model.quality_score * 0.6 + task_bonus,
                model.cost_score,
                model.name,
            )

        return sorted(candidates, key=score)

    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        """Text of the most recent user turn, for classification.

        Handles both plain-string content and OpenAI multi-part content
        (lists of {"type": "text", ...} blocks), as sent by agent clients.
        """
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
        return ""

    def chat(
        self,
        prompt: str = "",
        *,
        system: str = DEFAULT_SYSTEM,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        dry_run: bool = False,
        stream: bool = False,
        history: list[dict] | None = None,
        session_id: str | None = None,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: "str | dict | None" = None,
    ) -> "dict | StreamResult":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Create a .env file and paste your key."
            )

        if messages is not None:
            # Serving mode (e.g. server.py): the client sends a full OpenAI
            # message list, including its own system prompt and tool results.
            # Use it verbatim; classify on the latest user turn.
            messages = list(messages)
            if not prompt:
                prompt = self._last_user_text(messages)
        else:
            # CLI/library mode: system prompt, prior turns, then this prompt.
            messages = [{"role": "system", "content": system}]
            if history:
                messages.extend(
                    msg for msg in history if msg.get("role") != "system"
                )
            messages.append({"role": "user", "content": prompt})

        task = self.classify(prompt, messages)
        route = self.route(prompt, messages, require_tools=tools is not None)

        if not route:
            allowed = ", ".join(sorted(self.allow_families or [])) or "all"
            raise RuntimeError(f"No model fits this request. Allowed families: {allowed}")

        route = self._apply_sticky(session_id, route)

        if dry_run:
            return {
                "task": task,
                "mode": self.mode,
                "session_id": session_id,
                "sticky": bool(session_id) and session_id in self._sticky_choices,
                "selected": route[0].name,
                "selected_family": route[0].family,
                "selected_model": route[0].model,
                "fallback_order": [m.model for m in route],
            }

        if stream:
            return StreamResult(
                self._stream_chat(
                    api_key=api_key,
                    route=route,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task=task,
                    session_id=session_id,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            )

        errors: list[str] = []

        for model in route:
            for attempt in range(1, self.max_retries_per_model + 1):
                try:
                    result = self._call_openrouter(
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        task=task,
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                    self._remember_sticky(session_id, model.name)
                    return result
                except Exception as exc:
                    errors.append(
                        f"{model.model} attempt {attempt}: {type(exc).__name__}: {exc}"
                    )

                    # Bad key/auth will fail for every model, so stop early.
                    if self._is_auth_error(exc):
                        raise RuntimeError(errors[-1]) from exc

                    # Non-retryable (e.g. 400 bad request) -> try the next model
                    # immediately rather than retrying the same one.
                    if not self._is_retryable(exc):
                        break

                    self._sleep_backoff(attempt, exc)

        raise RuntimeError("All routed models failed:\n" + "\n".join(errors[-12:]))

    @staticmethod
    def _build_headers() -> dict[str, str]:
        headers: dict[str, str] = {}
        if os.getenv("OPENROUTER_REFERER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER", "")
        if os.getenv("OPENROUTER_APP_NAME"):
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "")
        return headers

    def _build_client(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency. Run: pip install openai python-dotenv") from exc

        return OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=self._build_headers() or None,
            timeout=90.0,
        )

    @staticmethod
    def _reasoning_body(model: ModelProfile, task: TaskKind) -> dict:
        # Only ask for reasoning where it helps.
        if model.reasoning and task in {"coding", "reasoning", "long_context"}:
            return {"reasoning": {"enabled": True}}
        return {}

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        return type(exc).__name__ in {"AuthenticationError", "PermissionDeniedError"} or (
            getattr(exc, "status_code", None) in {401, 403}
        )

    @staticmethod
    def _is_bad_request(exc: Exception) -> bool:
        return type(exc).__name__ == "BadRequestError" or getattr(exc, "status_code", None) == 400

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if getattr(exc, "status_code", None) in {408, 409, 429, 500, 502, 503, 504}:
            return True
        return type(exc).__name__ in {
            "RateLimitError",
            "InternalServerError",
            "APITimeoutError",
            "APIConnectionError",
        }

    def _create(self, client, *, model, messages, temperature, max_tokens, extra_body, **kwargs):
        """Create a completion, retrying once without extra_body if a provider
        rejects an unknown field (e.g. the `reasoning` flag) with a 400."""
        try:
            return client.chat.completions.create(
                model=model.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
                **kwargs,
            )
        except Exception as exc:
            if extra_body and self._is_bad_request(exc):
                return client.chat.completions.create(
                    model=model.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=None,
                    **kwargs,
                )
            raise

    @staticmethod
    def _tool_kwargs(tools: list[dict] | None, tool_choice) -> dict:
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return kwargs

    def _call_openrouter(
        self,
        *,
        api_key: str,
        model: ModelProfile,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        task: TaskKind,
        tools: list[dict] | None = None,
        tool_choice=None,
    ) -> dict:
        client = self._build_client(api_key)
        extra_body = self._reasoning_body(model, task)

        response = self._create(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **self._tool_kwargs(tools, tool_choice),
        )

        choice = response.choices[0]
        tool_calls = [tc.model_dump() for tc in (choice.message.tool_calls or [])]

        return {
            "family": model.family,
            "model_name": model.name,
            "model": model.model,
            "content": choice.message.content or "",
            "tool_calls": tool_calls or None,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    def _stream_chat(
        self,
        *,
        api_key: str,
        route: list[ModelProfile],
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        task: TaskKind,
        session_id: str | None = None,
        tools: list[dict] | None = None,
        tool_choice=None,
    ):
        errors: list[str] = []

        for model in route:
            for attempt in range(1, self.max_retries_per_model + 1):
                content_started = False
                try:
                    gen = self._call_openrouter_stream(
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        task=task,
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                    while True:
                        try:
                            chunk = next(gen)
                        except StopIteration as stop:
                            self._remember_sticky(session_id, model.name)
                            return stop.value or {}
                        content_started = True
                        yield chunk
                except Exception as exc:
                    if content_started:
                        raise RuntimeError(
                            f"Stream interrupted: {type(exc).__name__}: {exc}"
                        ) from exc
                    errors.append(
                        f"{model.model} attempt {attempt}: {type(exc).__name__}: {exc}"
                    )
                    if self._is_auth_error(exc):
                        raise RuntimeError(errors[-1]) from exc
                    if not self._is_retryable(exc):
                        break
                    self._sleep_backoff(attempt, exc)

        raise RuntimeError("All routed models failed:\n" + "\n".join(errors[-12:]))

    @staticmethod
    def _merge_tool_call_delta(acc: dict[int, dict], tc) -> None:
        """Fold one streamed tool-call delta into the accumulator.

        OpenAI-style streams send tool calls in pieces keyed by ``index``:
        the id/name arrive first, then the JSON arguments in fragments.
        """
        index = getattr(tc, "index", 0) or 0
        entry = acc.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if getattr(tc, "id", None):
            entry["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                entry["function"]["name"] = fn.name
            if getattr(fn, "arguments", None):
                entry["function"]["arguments"] += fn.arguments

    def _call_openrouter_stream(
        self,
        *,
        api_key: str,
        model: ModelProfile,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        task: TaskKind,
        tools: list[dict] | None = None,
        tool_choice=None,
    ):
        client = self._build_client(api_key)
        extra_body = self._reasoning_body(model, task)

        response = self._create(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
            **self._tool_kwargs(tools, tool_choice),
        )

        content_parts: list[str] = []
        finish_reason: str | None = None
        usage: dict | None = None
        tool_calls_acc: dict[int, dict] = {}

        for event in response:
            if event.usage:
                usage = event.usage.model_dump()
            if not event.choices:
                continue
            delta = event.choices[0]
            if delta.finish_reason:
                finish_reason = delta.finish_reason
            if delta.delta and delta.delta.tool_calls:
                for tc in delta.delta.tool_calls:
                    self._merge_tool_call_delta(tool_calls_acc, tc)
            if delta.delta and delta.delta.content:
                content_parts.append(delta.delta.content)
                yield delta.delta.content

        return {
            "family": model.family,
            "model_name": model.name,
            "model": model.model,
            "content": "".join(content_parts),
            "tool_calls": [tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None,
            "finish_reason": finish_reason,
            "usage": usage,
        }

    @staticmethod
    def _sleep_backoff(attempt: int, exc: Exception | None = None) -> None:
        delay: float | None = None

        # Honor a Retry-After header when the provider sends one (common on 429).
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    delay = float(retry_after)
            except (AttributeError, ValueError, TypeError):
                delay = None

        if delay is None:
            delay = (2 ** (attempt - 1)) + random.random()

        time.sleep(min(30.0, delay))

    def validate_models(self, api_key: str) -> dict:
        """Check configured model slugs against OpenRouter's live catalog.

        Model slugs drift over time; this flags any that no longer exist so the
        MODELS table can be updated.
        """
        client = self._build_client(api_key)
        available = {m.id for m in client.models.list().data}
        configured = {m.model for m in self.models}
        missing = sorted(configured - available)
        return {
            "available_count": len(available),
            "configured_count": len(configured),
            "missing": missing,
            "ok": not missing,
        }


def print_known_models() -> None:
    rows = [
        {
            "name": m.name,
            "family": m.family,
            "model": m.model,
            "cost_score": m.cost_score,
            "quality_score": m.quality_score,
            "context": m.max_context_tokens,
            "strengths": sorted(m.strengths),
        }
        for m in MODELS
    ]

    print(json.dumps(rows, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-key router for cheap Chinese LLMs via OpenRouter"
    )

    parser.add_argument("--prompt", help="Prompt to send")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument(
        "--mode",
        choices=["auto", "cheap", "balanced", "quality"],
        default="auto",
    )
    parser.add_argument(
        "--family",
        action="append",
        help="Restrict to a family: deepseek, qwen, kimi, glm, minimax, mimo. Can be repeated.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Only show routing decision")
    parser.add_argument("--list-models", action="store_true", help="Print the built-in model table")
    parser.add_argument("--stream", action="store_true", help="Stream output as it arrives")
    parser.add_argument(
        "--validate-models",
        action="store_true",
        help="Check configured model slugs against OpenRouter's live catalog",
    )
    parser.add_argument(
        "--history",
        help="Path to a JSON file with prior messages [{'role','content'}, ...]",
    )
    parser.add_argument(
        "--session-id",
        help=(
            "Pin a session to its first-chosen model so follow-up turns reuse it "
            "(keeps the provider prompt cache warm). Persisted across CLI runs."
        ),
    )

    args = parser.parse_args()

    if args.list_models:
        print_known_models()
        return 0

    if args.validate_models:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("OPENROUTER_API_KEY is missing.", file=sys.stderr)
            return 1
        try:
            report = EasyChineseModelRouter().validate_models(api_key)
        except Exception as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if not args.prompt:
        print(
            'Missing --prompt. Example: python router.py --prompt "fix this Python bug"',
            file=sys.stderr,
        )
        return 2

    # Persist session->model pins to a temp file so stickiness survives across
    # separate CLI invocations (each run is a fresh process).
    sticky_store = (
        os.path.join(tempfile.gettempdir(), "echinese_router_sessions.json")
        if args.session_id
        else None
    )

    router = EasyChineseModelRouter(
        mode=args.mode,
        allow_families=set(args.family) if args.family else None,
        sticky_store=sticky_store,
    )

    history: list[dict] | None = None
    if args.history:
        try:
            with open(args.history, "r", encoding="utf-8") as fh:
                history = json.load(fh)
            if not isinstance(history, list):
                raise ValueError("history file must contain a JSON list of messages")
        except Exception as exc:
            print(f"Failed to load history: {exc}", file=sys.stderr)
            return 1

    try:
        if args.stream and not args.dry_run:
            stream_result = router.chat(
                args.prompt,
                system=args.system,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
                history=history,
                session_id=args.session_id,
            )
            for chunk in stream_result:
                print(chunk, end="", flush=True)
            meta = stream_result.meta
            print(
                f"\n---\nfamily={meta.get('family')} model={meta.get('model')} usage={meta.get('usage')}",
                file=sys.stderr,
            )
            return 0

        result = router.chat(
            args.prompt,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            history=history,
            session_id=args.session_id,
        )
    except Exception as exc:
        print(f"Router failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        print(result["content"])
        print(
            f"\n---\nfamily={result['family']} model={result['model']} usage={result['usage']}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())