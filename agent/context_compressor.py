"""长会话的默认上下文压缩引擎。

核心策略：保留头部和最近尾部消息，把中间较旧的消息压缩为结构化 handoff
summary。摘要通常由 auxiliary compression 模型生成；失败时可按配置选择中止
压缩或生成本地 deterministic fallback summary。

这个文件解决的不只是“摘要”：还包括旧工具输出裁剪、多模态图片裁剪、tool_call
和 tool_result 配对修复、重复摘要迭代更新、失败冷却、抗空转压缩等上下文工程细节。
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, _is_connection_error
from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "If the latest user message is consistent with the '## Active Task' "
    "section, you may use the summary as background. If the latest user "
    "message contradicts, supersedes, changes topic from, or in any way "
    "diverges from '## Active Task' / '## In Progress' / '## Pending User "
    "Asks' / '## Remaining Work', the latest message WINS — discard those "
    "stale items entirely and do not 'wrap up the old task first'. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# Handoff prefixes that shipped in earlier releases. A summary persisted under
# one of these can be inherited into a resumed lineage (#35344); when it is
# re-normalized on re-compaction we must strip the OLD prefix too, otherwise the
# stale directive it carried (e.g. "resume exactly from Active Task") survives
# embedded in the body and keeps hijacking replies. Keep newest-first; entries
# are matched literally. Add a frozen copy here whenever SUMMARY_PREFIX changes.
_HISTORICAL_SUMMARY_PREFIXES = (
    # Pre-#35344: contained the self-contradicting "resume exactly" directive.
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:",
)

# Minimum tokens for the summary output
_MIN_SUMMARY_TOKENS = 2000
# Proportion of compressed content to allocate for summary
_SUMMARY_RATIO = 0.20
# Absolute ceiling for summary tokens (even on very large context windows)
_SUMMARY_TOKENS_CEILING = 12_000

# Placeholder used when pruning old tool results
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Chars per token rough estimate
_CHARS_PER_TOKEN = 4
# Flat token cost per attached image part.  Real cost varies by provider and
# dimensions (Anthropic ≈ width×height/750, GPT-4o up to ~1700 for
# high-detail 2048×2048, Gemini 258/tile), but 1600 is a realistic ceiling
# that keeps compression budgeting honest for multi-image conversations.
# Matches Claude Code's IMAGE_TOKEN_ESTIMATE constant.
_IMAGE_TOKEN_ESTIMATE = 1600
# Same figure expressed in the char-budget currency the rest of the
# compressor speaks in.  Used when accumulating message "content length"
# for tail-cut decisions.
_IMAGE_CHAR_EQUIVALENT = _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600

# Hard ceiling for the deterministic summary-failure handoff.  The fallback is
# only meant to preserve continuity anchors from the dropped window, not to
# become another unbounded transcript copy after the LLM summarizer failed.
_FALLBACK_SUMMARY_MAX_CHARS = 8_000
_FALLBACK_TURN_MAX_CHARS = 700


_PATH_MENTION_RE = re.compile(r"(?:/|~/?|[A-Za-z]:\\)[^\s`'\")\]}<>]+")


def _dedupe_append(items: list[str], value: str, *, limit: int) -> None:
    """向列表追加去重后的非空值，并限制最大数量。"""
    value = value.strip()
    if value and value not in items and len(items) < limit:
        items.append(value)


def _extract_tool_call_name_and_args(tool_call: Any) -> tuple[str, str]:
    """从 dict 或对象形态的 tool_call 中提取名称和参数。"""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        return str(fn.get("name") or "unknown"), str(fn.get("arguments") or "")

    fn = getattr(tool_call, "function", None)
    if fn is None:
        return "unknown", ""
    return str(getattr(fn, "name", None) or "unknown"), str(getattr(fn, "arguments", None) or "")


def _extract_tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(getattr(tool_call, "id", "") or "")


def _collect_path_mentions(text: str, relevant_files: list[str], *, limit: int = 12) -> None:
    for match in _PATH_MENTION_RE.findall(text):
        _dedupe_append(relevant_files, match.rstrip(".,:;"), limit=limit)


def _content_length_for_budget(raw_content: Any) -> int:
    """估算消息内容在压缩预算中的等效字符长度，图片按固定 token 成本折算。"""
    if isinstance(raw_content, str):
        return len(raw_content)
    if not isinstance(raw_content, list):
        return len(str(raw_content or ""))

    total = 0
    for p in raw_content:
        if isinstance(p, str):
            total += len(p)
            continue
        if not isinstance(p, dict):
            total += len(str(p))
            continue
        ptype = p.get("type")
        if ptype in {"image_url", "input_image", "image"}:
            total += _IMAGE_CHAR_EQUIVALENT
        else:
            # text / input_text / tool_result-with-text / anything else with
            # a text field.  Ignore the raw base64 payload inside image_url
            # dicts — dimensions don't matter, only whether it's an image.
            total += len(p.get("text", "") or "")
    return total


def _content_text_for_contains(content: Any) -> str:
    """把多种 content 形态投影成文本，仅用于包含关系判断。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _append_text_to_content(content: Any, text: str, *, prepend: bool = False) -> Any:
    """安全地向字符串或多模态 content 追加/前置文本块。"""
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text


def _strip_image_parts_from_parts(parts: Any) -> Any:
    """把旧 content parts 里的图片替换成文本占位符，用于裁剪历史截图。"""
    if not isinstance(parts, list):
        return None
    had_image = False
    out = []
    for part in parts:
        if not isinstance(part, dict):
            out.append(part)
            continue
        ptype = part.get("type")
        if ptype in {"image", "image_url", "input_image"}:
            had_image = True
            out.append({"type": "text", "text": "[screenshot removed to save context]"})
        else:
            out.append(part)
    return out if had_image else None


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """在保持 JSON 合法性的前提下，截短 tool_call 参数里的长字符串。"""
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    shrunken = _shrink(parsed)
    # ensure_ascii=False preserves CJK/emoji instead of bloating with \uXXXX
    return json.dumps(shrunken, ensure_ascii=False)


_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _is_image_part(part: Any) -> bool:
    """判断 content part 是否是 OpenAI/Responses/Anthropic 任一形态的图片块。"""
    if not isinstance(part, dict):
        return False
    return part.get("type") in _IMAGE_PART_TYPES


def _content_has_images(content: Any) -> bool:
    """判断消息 content 是否包含图片块。"""
    if not isinstance(content, list):
        return False
    return any(_is_image_part(p) for p in content)


def _strip_images_from_content(content: Any) -> Any:
    """返回图片被文本占位符替换后的 content 副本，不修改入参。"""
    if not isinstance(content, list):
        return content
    if not any(_is_image_part(p) for p in content):
        return content

    new_parts: List[Any] = []
    for p in content:
        if _is_image_part(p):
            new_parts.append({
                "type": "text",
                "text": "[Attached image — stripped after compression]",
            })
        else:
            new_parts.append(p)
    return new_parts


def _strip_historical_media(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """裁掉最新含图用户消息之前的历史图片，避免反复发送大 base64 payload。"""
    if not messages:
        return messages

    # Find the newest user message that carries at least one image part.
    # We anchor on image-bearing user messages (not all user messages) so
    # a plain text follow-up after a big-image turn still strips the old
    # image — matching the problem kilocode#9434 set out to solve.
    anchor = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        if _content_has_images(msg.get("content")):
            anchor = i
            break

    if anchor <= 0:
        # No image-bearing user message, or it's the very first message —
        # nothing before it to strip.
        return messages

    changed = False
    result: List[Dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= anchor or not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if not _content_has_images(content):
            result.append(msg)
            continue
        new_msg = msg.copy()
        new_msg["content"] = _strip_images_from_content(content)
        result.append(new_msg)
        changed = True

    return result if changed else messages


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """把旧工具结果压成一行可读摘要，保留命令、路径、结果规模等关键信息。"""
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in {"browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"}:
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in {"skill_view", "skills_list", "skill_manage"}:
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


class ContextCompressor(ContextEngine):
    """默认上下文引擎：用有损摘要压缩长会话。"""

    @property
    def name(self) -> str:
        return "compressor"

    def on_session_reset(self) -> None:
        """``/new`` 或 ``/reset`` 时清空压缩器的会话级状态。"""
        super().on_session_reset()
        self._context_probed = False
        self._context_probe_persistable = False
        self._previous_summary = None
        self._last_summary_error = None
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._summary_failure_cooldown_until = 0.0  # transient errors must not block a fresh session
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """真实会话结束时清掉迭代摘要，防止跨 session 继承旧 summary。"""
        self._previous_summary = None

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: Any = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """模型切换或 fallback 后刷新上下文窗口、阈值和尾部保护预算。"""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = max(
            int(context_length * self.threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        # Recalculate token budgets for the new context length so the
        # compressor stays calibrated after a model switch (e.g. 200K → 32K).
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
        abort_on_summary_failure: bool = False,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode
        # When True, summary-generation failure aborts compression entirely
        # (returns messages unchanged, sets _last_compress_aborted=True).
        # When False (default = historical behavior), insert a
        # deterministic "summary unavailable" handoff and drop the middle window.
        self.abort_on_summary_failure = abort_on_summary_failure

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        # Floor: never compress below MINIMUM_CONTEXT_LENGTH tokens even if
        # the percentage would suggest a lower value.  This prevents premature
        # compression on large-context models at 50% while keeping the % sane
        # for models right at the minimum.
        self.threshold_tokens = max(
            int(self.context_length * threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        self.compression_count = 0

        # Derive token budgets: ratio is relative to the threshold, not total context
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        self._context_probed = False  # True after a step-down from context error

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

        self.summary_model = summary_model_override or ""

        # Stores the previous compaction summary for iterative updates
        self._previous_summary: Optional[str] = None
        # Anti-thrashing: track whether last compression was effective
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        self._summary_failure_cooldown_until: float = 0.0
        self._last_summary_error: Optional[str] = None
        # When summary generation fails and a static fallback is inserted,
        # record how many turns were unrecoverably dropped so callers
        # (gateway hygiene, /compress) can surface a visible warning.
        self._last_summary_dropped_count: int = 0
        self._last_summary_fallback_used: bool = False
        # When summary generation fails we now ABORT compression entirely
        # and return the original messages unchanged instead of dropping
        # the middle window with a static placeholder.  Callers inspect
        # this flag to know "compression was attempted but aborted, freeze
        # the chat until the user manually retries via /compress".
        self._last_compress_aborted: bool = False
        # When a user-configured summary model fails and we recover by
        # retrying on the main model, record the failure so gateway /
        # CLI callers can still warn the user even though compression
        # succeeded.  Silent recovery would hide the broken config.
        self._last_aux_model_failure_error: Optional[str] = None
        self._last_aux_model_failure_model: Optional[str] = None

    def update_from_response(self, usage: Dict[str, Any]):
        """从 provider 的真实 usage 更新 token 状态，供后续压缩判断使用。"""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens)
        if self.last_prompt_tokens > 0:
            self.last_real_prompt_tokens = self.last_prompt_tokens
            if self.last_prompt_tokens < self.threshold_tokens:
                if self.awaiting_real_usage_after_compression and self.last_compression_rough_tokens > 0:
                    self.last_rough_tokens_when_real_prompt_fit = self.last_compression_rough_tokens
            else:
                self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """粗略估算受工具 schema 噪声影响时，优先相信最近真实 prompt_tokens。"""
        if rough_tokens < self.threshold_tokens:
            return False
        if self.last_real_prompt_tokens <= 0:
            return False
        if self.last_real_prompt_tokens >= self.threshold_tokens:
            return False

        baseline = self.last_rough_tokens_when_real_prompt_fit or self.last_compression_rough_tokens
        if baseline <= 0:
            return False

        growth = max(0, rough_tokens - baseline)
        tolerated_growth = max(4096, int(self.threshold_tokens * 0.05))
        if growth > tolerated_growth:
            return False

        self.last_rough_tokens_when_real_prompt_fit = max(baseline, rough_tokens)
        return True

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """判断是否超过压缩阈值，并用低收益计数避免压缩空转。"""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compressions saved <10%% each. "
                    "Consider /new to start a fresh session, or /compress <topic> "
                    "for focused compression.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Tool output pruning (cheap pre-pass, no LLM call)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """裁剪旧工具结果：去重、压成一行摘要，并截短旧 tool_call 参数。"""
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # 建立 tool_call_id 到工具名/参数的索引，后面摘要 tool result 时要用。
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # 计算裁剪边界：优先按 token 预算保护尾部，消息数量作为最低保护线。
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            # 从尾部向前累加 token，预算内的消息保留完整。
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result))
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                raw_content = msg.get("content") or ""
                content_len = _content_length_for_budget(raw_content)
                msg_tokens = content_len // _CHARS_PER_TOKEN + 10
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        args = tc.get("function", {}).get("arguments", "")
                        msg_tokens += len(args) // _CHARS_PER_TOKEN
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            # Translate the budget walk into a "protected count", apply the
            # floor in count-space (where `max` reads naturally: protect at
            # least `min_protect` messages or whatever the budget reserved,
            # whichever is more), then convert back to a prune boundary.
            # Doing this in index-space with `max` would invert the direction
            # (smaller index = MORE protected), so a generous budget would
            # silently get truncated back down to `min_protect`.
            budget_protect_count = len(result) - boundary
            protected_count = max(budget_protect_count, min_protect)
            prune_boundary = len(result) - protected_count
        else:
            prune_boundary = len(result) - protect_tail_count

        # 第一遍：重复的大型 tool result 只保留最新完整副本。
        content_hashes: dict = {}  # hash -> (index, tool_call_id)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Multimodal content — dedupe by the text summary if available.
            if isinstance(content, list):
                continue
            if not isinstance(content, str):
                # Multimodal dict envelopes ({_multimodal: True, content: [...]}) and
                # other non-string tool-result shapes can't be hashed/deduped by text.
                continue
            if len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                # 这是旧重复结果，替换成指向较新结果的占位说明。
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # 第二遍：把保护区之外的大型 tool result 替换成信息摘要。
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # 多模态工具输出里的截图很大，只保留轻量文本占位符。
            if isinstance(content, list):
                stripped = _strip_image_parts_from_parts(content)
                if stripped is not None:
                    result[i] = {**msg, "content": stripped}
                    pruned += 1
                continue
            if isinstance(content, dict) and content.get("_multimodal"):
                summary = content.get("text_summary") or "[screenshot removed to save context]"
                result[i] = {**msg, "content": f"[screenshot removed] {summary[:200]}"}
                pruned += 1
                continue
            if not isinstance(content, str):
                continue
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            # 已去重或已摘要过的结果不再重复处理。
            if content.startswith("[Duplicate tool output"):
                continue
            # 只有较长内容才值得摘要，短结果保持原样。
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # 第三遍：截短旧 assistant tool_call 参数，避免 write_file 等参数撑爆上下文。
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_call_args_json(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """按被压缩内容规模计算摘要 token 预算，并受模型上下文窗口上限约束。"""
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    # 摘要模型输入截断上限：控制每条消息喂给 summary model 的最大字符量。
    _CONTENT_MAX = 6000       # total chars per message body
    _CONTENT_HEAD = 4000      # chars kept from the start
    _CONTENT_TAIL = 1500      # chars kept from the end
    _TOOL_ARGS_MAX = 1500     # tool call argument chars
    _TOOL_ARGS_HEAD = 1200    # kept from the start of tool args

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """把待压缩消息序列化为带角色标签的文本，并先做敏感信息脱敏。"""
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = redact_sensitive_text(msg.get("content") or "")

            # 工具结果保留足够内容，让摘要能记住路径、命令和错误。
            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            # assistant 消息需要把工具名和参数一起给摘要模型。
            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = redact_sensitive_text(fn.get("arguments", ""))
                            # 长参数只保留头部，避免 summary prompt 过大。
                            if len(args) > self._TOOL_ARGS_MAX:
                                args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            # 用户和其他角色按普通文本处理。
            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def _build_static_fallback_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        reason: str | None = None,
    ) -> str:
        """LLM 摘要不可用时，本地生成结构化 fallback handoff。"""
        user_asks: list[str] = []
        assistant_actions: list[str] = []
        tool_actions: list[str] = []
        relevant_files: list[str] = []
        blockers: list[str] = []
        last_dropped_turns: list[str] = []

        def _compact_fallback_turn(value: Any) -> str:
            text = redact_sensitive_text(_content_text_for_contains(value))
            text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > _FALLBACK_TURN_MAX_CHARS:
                text = text[: _FALLBACK_TURN_MAX_CHARS - 15].rstrip() + " ...[truncated]"
            return re.sub(r"\bgh[pousr]_[A-Za-z0-9_.-]+", "[REDACTED]", text)

        def _remember_dropped_turn(label: str, text: str, *, limit: int = 8) -> None:
            text = text.strip()
            if not text:
                return
            last_dropped_turns.append(f"{label}: {text}")
            if len(last_dropped_turns) > limit:
                del last_dropped_turns[0]

        def _collect_paths_from_jsonish(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key in {"path", "workdir", "file_path", "output_path"} and isinstance(val, str):
                        _dedupe_append(relevant_files, val, limit=12)
                    _collect_paths_from_jsonish(val)
            elif isinstance(obj, list):
                for val in obj:
                    _collect_paths_from_jsonish(val)
            elif isinstance(obj, str):
                _collect_path_mentions(obj, relevant_files)

        call_id_to_tool: dict[str, tuple[str, str]] = {}
        for msg in turns_to_summarize:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    name, raw_args = _extract_tool_call_name_and_args(tc)
                    args = redact_sensitive_text(raw_args)
                    call_id = _extract_tool_call_id(tc)
                    if call_id:
                        call_id_to_tool[call_id] = (name, args)
                    if args:
                        try:
                            parsed = json.loads(args)
                        except Exception:
                            parsed = args
                        _collect_paths_from_jsonish(parsed)

        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            text = _compact_fallback_turn(msg.get("content"))
            _collect_path_mentions(text, relevant_files)

            turn_text = text
            turn_tool_names: list[str] = []
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls") or []:
                    name, _args = _extract_tool_call_name_and_args(tc)
                    turn_tool_names.append(name)
                if turn_tool_names:
                    prefix = "tool calls: " + ", ".join(turn_tool_names[:6])
                    turn_text = f"{prefix}; {turn_text}" if turn_text else prefix
            _remember_dropped_turn(str(role).upper(), turn_text)

            if len(text) > 600:
                text = text[:420].rstrip() + " ... " + text[-160:].lstrip()

            if role == "user" and text:
                user_asks.append(text)
            elif role == "assistant":
                tool_names: list[str] = []
                for tc in msg.get("tool_calls") or []:
                    name, _args = _extract_tool_call_name_and_args(tc)
                    tool_names.append(name)
                if tool_names:
                    assistant_actions.append(
                        "Called tool(s): " + ", ".join(tool_names[:6])
                    )
                elif text:
                    assistant_actions.append(text)
            elif role == "tool":
                call_id = str(msg.get("tool_call_id") or "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                tool_actions.append(
                    _summarize_tool_result(tool_name, tool_args, text or "")
                )
                if re.search(
                    r"\b(error|failed|exception|traceback|timeout|timed out|fatal)\b",
                    text,
                    re.I,
                ):
                    blockers.append(text[:500])

        def _bullets(items: list[str], limit: int = 8) -> str:
            unique: list[str] = []
            seen: set[str] = set()
            for item in items:
                item = item.strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                unique.append(item)
                if len(unique) >= limit:
                    break
            return "\n".join(f"- {item}" for item in unique) if unique else "None."

        completed: list[str] = []
        for idx, item in enumerate((assistant_actions + tool_actions)[:12], start=1):
            completed.append(f"{idx}. {item}")

        active_task = (
            f"User asked: {user_asks[-1]!r}"
            if user_asks
            else "Unknown from deterministic fallback."
        )
        previous_summary_note = ""
        if self._previous_summary:
            previous_summary_note = (
                "\n\nPrevious compaction summary was present and should still be treated as "
                "background continuity context, but the latest LLM summary update failed."
            )

        reason_text = f" Summary failure reason: {reason}." if reason else ""
        body = f"""## Active Task
{active_task}

## Goal
Recovered from a deterministic fallback because the LLM context summarizer was unavailable. Continue from the protected recent messages after this summary and use current file/system state for exact details.{previous_summary_note}

## Constraints & Preferences
- This fallback was generated locally without an LLM summary call.
- Secrets and credentials were redacted before preservation.
- The summary may be incomplete; prefer verifying current files, git state, processes, and test results instead of assuming omitted details.

## Completed Actions
{chr(10).join(completed) if completed else "None recoverable from compacted turns."}

## Active State
Unknown from deterministic fallback. Inspect current repository/session state if needed.

## In Progress
{active_task}

## Blocked
{_bullets(blockers, limit=5)}

## Key Decisions
None recoverable from deterministic fallback.

## Resolved Questions
None recoverable from deterministic fallback.

## Pending User Asks
{active_task}

## Relevant Files
{_bullets(relevant_files, limit=12)}

## Remaining Work
Continue from the most recent unfulfilled user ask and protected tail messages. Verify state with tools before making claims.

## Last Dropped Turns
{_bullets(last_dropped_turns, limit=8)}

## Critical Context
Summary generation was unavailable, so this is a best-effort deterministic fallback for {len(turns_to_summarize)} compacted message(s).{reason_text}"""
        summary = self._with_summary_prefix(redact_sensitive_text(body.strip()))
        if len(summary) > _FALLBACK_SUMMARY_MAX_CHARS:
            summary = summary[: _FALLBACK_SUMMARY_MAX_CHARS - 42].rstrip() + "\n...[fallback summary truncated]"
        return summary

    def _fallback_to_main_for_compression(self, e: Exception, reason: str) -> None:
        """summary_model 失败时切回主模型，并记录可展示的失败原因。"""
        self._summary_model_fallen_back = True
        logger.warning(
            "Summary model '%s' %s (%s). "
            "Falling back to main model '%s' for compression.",
            self.summary_model, reason, e, self.model,
        )
        _err_text = str(e).strip() or e.__class__.__name__
        if len(_err_text) > 220:
            _err_text = _err_text[:217].rstrip() + "..."
        self._last_aux_model_failure_error = _err_text
        self._last_aux_model_failure_model = self.summary_model
        self.summary_model = ""  # empty = use main model
        self._summary_failure_cooldown_until = 0.0  # no cooldown — retry immediately

    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> Optional[str]:
        """调用 auxiliary/main LLM 生成结构化摘要；有旧摘要时做迭代更新。"""
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - now,
            )
            return None

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)

        # 给摘要加日期锚点，避免“今天/刚才/继续做”在恢复后变成新指令。
        try:
            from hermes_time import now as _hermes_now

            _today_str = _hermes_now().strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - clock resolution is best-effort
            _today_str = ""

        # 首次压缩和迭代压缩共用的摘要模型前置说明。
        _summarizer_preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Treat the conversation turns below as source material for a "
            "compact record of prior work. "
            "Produce only the structured summary; do not add a greeting, "
            "preamble, or prefix. "
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that the user had credentials present, but "
            "do not preserve their values."
        )

        # 时间锚点规则：已完成动作要写成带日期的过去事实。
        if _today_str:
            _temporal_anchoring_rule = (
                f"\nTEMPORAL ANCHORING: The current date is {_today_str}. When an "
                "action has already been carried out, phrase it as a completed, "
                "dated, past-tense fact rather than an open instruction. For "
                'example, rewrite "email John about the proposal" as "Sent the '
                f'proposal email to John on {_today_str}." Never leave a finished '
                "action worded as if it still needs doing, and never invent a date "
                "for work that has not happened yet.\n"
            )
        else:
            _temporal_anchoring_rule = ""

        # 摘要固定结构：后续模型依赖这些 section 恢复任务状态。
        _template_sections = f"""## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Capture the user's most recent unfulfilled
input verbatim — the exact words they used. This includes:
- Explicit task assignments ("refactor the auth module")
- Questions awaiting an answer ("waarom staat X op Y?", "wat zijn de volgende stappen?")
- Decisions awaiting input ("optie A of B?")
- Ongoing discussions where the assistant owes the next substantive reply
A conversation where the user just asked a question IS an active task — the
task is "answer that question with full context". Do NOT write "None" merely
because the user did not issue an imperative command; reserve "None" for the
rare case where the last exchange was fully resolved and the user said
something like "thanks, that's all".
If multiple items are outstanding, list only the ones NOT yet completed.
Continuation should pick up exactly here. Examples:
"User asked: 'Now refactor the auth module to use JWT instead of sessions'"
"User asked: 'Waarom stond provider ineens op openrouter?' — needs investigation + answer"
"User chose option A; awaiting implementation of step 2"
If the user's most recent message was a reverse signal (stop, undo, roll
back, never mind, just verify, change of topic) that supersedes earlier
work, write the reverse signal verbatim and DO NOT carry forward the
cancelled task. Example: "User asked: 'Stop the i18n refactor and just
verify the current diff' — earlier i18n in-flight work is cancelled."
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Example:
1. READ config.py:45 — found `==` should be `!=` [tool: read_file]
2. PATCH config.py:45 — changed `==` to `!=` [tool: patch]
3. TEST `pytest tests/` — 3/50 failed: test_parse, test_validate, test_edge [tool: terminal]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include:
- Working directory and branch (if applicable)
- Modified/created files with brief note on each
- Test status (X/Y passing)
- Any running processes or servers
- Environment details that matter]

## In Progress
[Work currently underway — what was being done when compaction fired]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so it is not repeated]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not instructions]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.
{_temporal_anchoring_rule}
Write only the summary body. Do not include any preamble or prefix."""

        if self._previous_summary:
            # 迭代更新：保留仍然有效的旧摘要，再合并新进展。
            prompt = f"""{_summarizer_preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move items from "In Progress" to "Completed Actions" when done. Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete. CRITICAL: Update "## Active Task" to reflect the user's most recent unfulfilled input — this includes any question, decision request, or discussion turn that the assistant has not yet answered. Only write "None" if the last exchange was fully resolved.

{_template_sections}"""
        else:
            # 首次压缩：从待压缩 turns 直接生成摘要。
            prompt = f"""{_summarizer_preamble}

Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{_template_sections}"""

        # 手动 /compress <focus> 时，把焦点主题放在 prompt 末尾提高优先级。
        if focus_topic:
            prompt += f"""

FOCUS TOPIC: "{focus_topic}"
The user has requested that this compaction PRIORITISE preserving all information related to the focus topic above. For content related to "{focus_topic}", include full detail — exact values, file paths, command outputs, error messages, and decisions. For content NOT related to the focus topic, summarise more aggressively (brief one-liners or omit if truly irrelevant). The focus topic sections should receive roughly 60-70% of the summary token budget. Even for the focus topic, NEVER preserve API keys, tokens, passwords, or credentials — use [REDACTED]."""

        try:
            call_kwargs = {
                "task": "compression",
                "main_runtime": {
                    "model": self.model,
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "api_mode": self.api_mode,
                },
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(summary_budget * 1.3),
                # timeout resolved from auxiliary.compression.timeout config by call_llm
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            response = call_llm(**call_kwargs)
            content = response.choices[0].message.content
            # Handle cases where content is not a string (e.g., dict from llama.cpp)
            if not isinstance(content, str):
                content = str(content) if content else ""
            # Redact the summary output as well — the summarizer LLM may
            # ignore prompt instructions and echo back secrets verbatim.
            summary = redact_sensitive_text(content.strip())
            # Store for iterative updates on next compaction
            self._previous_summary = summary
            self._summary_failure_cooldown_until = 0.0
            self._summary_model_fallen_back = False
            self._last_summary_error = None
            return self._with_summary_prefix(summary)
        except RuntimeError:
            # No provider configured — long cooldown, unlikely to self-resolve
            self._summary_failure_cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            self._last_summary_error = "no auxiliary LLM provider configured"
            logger.warning("Context compression: no provider available for "
                            "summary. Middle turns will be dropped without summary "
                            "for %d seconds.",
                            _SUMMARY_FAILURE_COOLDOWN_SECONDS)
            return None
        except Exception as e:
            # If the summary model is different from the main model and the
            # error looks permanent (model not found, 503, 404), fall back to
            # using the main model instead of entering cooldown that leaves
            # context growing unbounded.  (#8620 sub-issue 4)
            _status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            _err_str = str(e).lower()
            _is_model_not_found = (
                _status in {404, 503}
                or "model_not_found" in _err_str
                or "does not exist" in _err_str
                or "no available channel" in _err_str
            )
            _is_timeout = (
                _status in {408, 429, 502, 504}
                or "timeout" in _err_str
            )
            # Non-JSON / malformed-body responses from misconfigured providers
            # or proxies (e.g. an HTML 502 page returned with
            # ``Content-Type: application/json``) bubble up as
            # ``json.JSONDecodeError`` from the OpenAI SDK's ``response.json()``,
            # or as a wrapping ``APIResponseValidationError`` whose message
            # carries the substring "expecting value".  Treat these like a
            # transient provider failure: one retry on the main model, then a
            # short cooldown.  Issue #22244.
            _is_json_decode = (
                isinstance(e, json.JSONDecodeError)
                or "expecting value" in _err_str
            )
            # httpcore / httpx streaming premature-close errors surface as
            # ConnectionError subclasses or plain Exception with characteristic
            # substrings ("incomplete chunked read", "peer closed connection",
            # "response ended prematurely", "unexpected eof").  These are
            # transient network events; treat them like a timeout so we fall
            # back to the main model instead of entering a 60-second cooldown.
            # See issue #18458.
            _is_streaming_closed = _is_connection_error(e)
            if _is_json_decode and not _is_model_not_found and not _is_timeout:
                logger.error(
                    "Context compression failed: auxiliary LLM returned a "
                    "non-JSON response. provider=%s summary_model=%s "
                    "main_model=%s base_url=%s err=%s",
                    self.provider or "auto",
                    self.summary_model or "(main)",
                    self.model,
                    self.base_url or "default",
                    e,
                )
            if (
                (_is_model_not_found or _is_timeout or _is_json_decode or _is_streaming_closed)
                and self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                if _is_json_decode:
                    _reason = "returned invalid JSON"
                elif _is_model_not_found:
                    _reason = "unavailable"
                elif _is_streaming_closed:
                    _reason = "closed stream prematurely"
                else:
                    _reason = "timed out"
                self._fallback_to_main_for_compression(e, _reason)
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)  # retry immediately

            # Unknown-error best-effort retry on main model.  Losing N turns of
            # context is almost always worse than one extra summary attempt, so
            # if we haven't already fallen back and the summary model differs
            # from the main model, try once more on main before entering
            # cooldown.  Errors that DID match _is_model_not_found above are
            # already handled by the fast-path retry; this branch catches
            # everything else (400s, provider-specific "no route" strings,
            # aggregator rejections, etc.) where auto-retry is still safer
            # than dropping the turns.
            if (
                self.summary_model
                and self.summary_model != self.model
                and not getattr(self, "_summary_model_fallen_back", False)
            ):
                self._fallback_to_main_for_compression(e, "failed")
                return self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

            # Transient errors (timeout, rate limit, network, JSON decode,
            # streaming premature-close) — shorter cooldown for JSON decode and
            # streaming-closed since those conditions can self-resolve quickly.
            _transient_cooldown = 30 if (_is_json_decode or _is_streaming_closed) else 60
            self._summary_failure_cooldown_until = time.monotonic() + _transient_cooldown
            err_text = str(e).strip() or e.__class__.__name__
            if len(err_text) > 220:
                err_text = err_text[:217].rstrip() + "..."
            self._last_summary_error = err_text
            logger.warning(
                "Failed to generate context summary: %s. "
                "Further summary attempts paused for %d seconds.",
                e,
                _transient_cooldown,
            )
            return None

    @staticmethod
    def _strip_summary_prefix(summary: str) -> str:
        """去掉当前、旧版和历史 handoff prefix，只保留摘要正文。"""
        text = (summary or "").strip()
        for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX, *_HISTORICAL_SUMMARY_PREFIXES):
            if text.startswith(prefix):
                return text[len(prefix):].lstrip()
        return text

    @classmethod
    def _with_summary_prefix(cls, summary: str) -> str:
        """把摘要正文规范化为当前 handoff 格式。"""
        text = cls._strip_summary_prefix(summary)
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        text = _content_text_for_contains(content).lstrip()
        if text.startswith(SUMMARY_PREFIX) or text.startswith(LEGACY_SUMMARY_PREFIX):
            return True
        return any(text.startswith(p) for p in _HISTORICAL_SUMMARY_PREFIXES)

    @classmethod
    def _find_latest_context_summary(
        cls,
        messages: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> tuple[Optional[int], str]:
        """在压缩窗口内寻找最新的 handoff summary。"""
        for idx in range(end - 1, start - 1, -1):
            content = messages[idx].get("content")
            if cls._is_context_summary_content(content):
                return idx, cls._strip_summary_prefix(_content_text_for_contains(content))
        return None, ""

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """从 dict 或对象形态的 tool_call 中提取调用 ID。"""
        if isinstance(tc, dict):
            return tc.get("call_id", "") or tc.get("id", "") or ""
        return getattr(tc, "call_id", "") or getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压缩后修复孤儿 tool_call/tool_result，保证 provider 能接受消息序列。"""
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 删除找不到对应 assistant tool_call 的 tool result。
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        # 给仍存在但结果被压掉的 tool_call 补占位 tool result。
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = self._get_tool_call_id(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self.quiet_mode:
                logger.info("Compression sanitizer: added %d stub tool result(s)", len(missing_results))

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """把压缩起点向后推过 tool result，避免从工具结果组中间开始压缩。"""
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _protect_head_size(self, messages: List[Dict[str, Any]]) -> int:
        """计算受保护头部消息数；system prompt 存在时总是隐式保护。"""
        head = 0
        if messages and messages[0].get("role") == "system":
            head = 1
        return head + self.protect_first_n

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """把压缩终点向前拉，避免切开 assistant tool_call 与后续 tool results。"""
        if idx <= 0 or idx >= len(messages):
            return idx
        # 向前跨过连续 tool results。
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        # 如果找到了父 assistant tool_call，把整个工具组放进摘要区。
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """返回 head_end 之后最后一条 user 消息下标。"""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def _ensure_last_user_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """保证最新用户消息留在受保护尾部，避免当前任务被摘要吞掉。"""
        last_user_idx = self._find_last_user_message_idx(messages, head_end)
        if last_user_idx < 0:
            # head 之后没有 user 消息，无需锚定。
            return cut_idx

        if last_user_idx >= cut_idx:
            # 已经在尾部保护区，无需处理。
            return cut_idx

        # 最新 user 落进了摘要区，直接把尾部起点拉回该消息。
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last user message at index %d "
                "(was %d) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx,
            )
        # 安全兜底：不要把切点拉回受保护头部。
        return max(last_user_idx, head_end + 1)

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """从尾部按 token 预算寻找尾部保护区起点，并保持工具组/最新用户消息完整。"""
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        # 硬性最低保护：尾部至少保留 3 条消息。
        min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n  # start from beyond the end

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            raw_content = msg.get("content") or ""
            content_len = _content_length_for_budget(raw_content)
            msg_tokens = content_len // _CHARS_PER_TOKEN + 10  # +10 for role/metadata
            # tool_call 参数也会占上下文窗口。
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            # 超过软上限且已经满足最低尾部条数时停止。
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        # 如果整个 transcript 都在软上限内，用原始预算再算一次，避免压缩空转。
        if cut_idx <= head_end and accumulated <= soft_ceiling and accumulated > 0:
            # 重新用未放大的预算寻找可压缩的中间区。
            raw_budget = token_budget
            raw_accumulated = 0
            for j in range(n - 1, head_end - 1, -1):
                raw_msg = messages[j]
                raw_content = raw_msg.get("content") or ""
                raw_len = _content_length_for_budget(raw_content)
                raw_tok = raw_len // _CHARS_PER_TOKEN + 10
                for tc in raw_msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        args = tc.get("function", {}).get("arguments", "")
                        raw_tok += len(args) // _CHARS_PER_TOKEN
                if raw_accumulated + raw_tok > raw_budget and (n - j) >= min_tail:
                    cut_idx = j
                    break
                raw_accumulated += raw_tok
                cut_idx = j
            # 如果仍然全吃下，下面的 fallback 会强制切到 head 后。

        # 确保尾部最低消息条数。
        fallback_cut = n - min_tail
        cut_idx = min(cut_idx, fallback_cut)

        # 如果预算会保护全部消息，强制在 head 后切一刀以便真正压缩。
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        # 对齐边界，避免拆开工具调用组。
        cut_idx = self._align_boundary_backward(messages, cut_idx)

        # 最新用户消息必须在尾部保护区，当前任务不能落进摘要区。
        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # ContextEngine: manual /compress preflight
    # ------------------------------------------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """判断是否存在非空中间区，供 gateway ``/compress`` 预检使用。"""
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        return compress_start < compress_end

    # ------------------------------------------------------------------
    # Main compression entry point
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None, force: bool = False) -> List[Dict[str, Any]]:
        """压缩消息：保留头尾，把中间 turns 变成结构化 handoff summary。"""
        # 每次压缩前重置失败状态；调用方会根据这些字段决定是否提示用户。
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._last_compress_aborted = False

        # 手动 /compress 允许跳过失败冷却，方便用户立即重试。
        if force and self._summary_failure_cooldown_until > 0.0:
            self._summary_failure_cooldown_until = 0.0
        n_messages = len(messages)
        # 至少需要 head + 3 条尾部消息 + 可压缩中间区。
        _min_for_compress = self._protect_head_size(messages) + 3 + 1
        if n_messages <= _min_for_compress:
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need > %d)",
                    n_messages, _min_for_compress,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # 阶段 1：先本地裁剪旧工具结果，不消耗 LLM 调用。
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # 阶段 2：计算受保护头部和受保护尾部的边界。
        compress_start = self._protect_head_size(messages)
        compress_start = self._align_boundary_forward(messages, compress_start)

        # 尾部保护按 token 预算而不是固定消息数。
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            # 没有可压缩窗口时记录为低收益，避免后续反复 no-op 压缩。
            self._ineffective_compression_count += 1
            self._last_compression_savings_pct = 0.0
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped: compress_start (%d) >= compress_end (%d) "
                    "— transcript fits within tail budget, nothing to compress. "
                    "ineffective_compression_count=%d",
                    compress_start, compress_end,
                    self._ineffective_compression_count,
                )
            return messages

        turns_to_summarize = messages[compress_start:compress_end]
        # resume 后旧 handoff summary 可能还在 head 中；用它恢复迭代摘要状态。
        summary_search_start = 1 if messages and messages[0].get("role") == "system" else 0
        summary_idx, summary_body = self._find_latest_context_summary(
            messages,
            summary_search_start,
            compress_end,
        )
        if summary_idx is not None:
            if summary_body and not self._previous_summary:
                self._previous_summary = summary_body
            turns_to_summarize = messages[max(compress_start, summary_idx + 1):compress_end]
        elif self._previous_summary:
            # 当前消息没有 summary 但内存里有旧 summary，说明跨 session 残留，丢弃。
            self._previous_summary = None

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # 阶段 3：生成结构化摘要。
        summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

        # 摘要失败后的行为由 compression.abort_on_summary_failure 控制。
        if not summary and self.abort_on_summary_failure:
            n_skipped = compress_end - compress_start
            self._last_summary_dropped_count = 0  # nothing actually dropped
            self._last_summary_fallback_used = False
            self._last_compress_aborted = True
            if not self.quiet_mode:
                logger.warning(
                    "Summary generation failed — aborting compression "
                    "(compression.abort_on_summary_failure=true). "
                    "%d message(s) preserved unchanged. Conversation is "
                    "frozen until the next /compress or /new.",
                    n_skipped,
                )
            return messages

        # 阶段 4：拼装压缩后的消息列表。
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                _compression_note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work. Your persistent memory (MEMORY.md, USER.md) remains fully authoritative regardless of compaction.]"
                if _compression_note not in _content_text_for_contains(existing):
                    msg["content"] = _append_text_to_content(
                        existing,
                        "\n\n" + _compression_note if isinstance(existing, str) and existing else _compression_note,
                    )
            compressed.append(msg)

        # LLM 摘要失败时插入本地 fallback，保留最少可恢复锚点。
        if not summary:
            if not self.quiet_mode:
                logger.warning("Summary generation failed — inserting deterministic fallback context summary")
            n_dropped = compress_end - compress_start
            self._last_summary_dropped_count = n_dropped
            self._last_summary_fallback_used = True
            summary = self._build_static_fallback_summary(
                turns_to_summarize,
                reason=self._last_summary_error,
            )

        _merge_summary_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
        # 选择 summary 消息角色，尽量避免相邻同角色破坏 provider 消息格式。
        if last_head_role in {"assistant", "tool"}:
            summary_role = "user"
        else:
            summary_role = "assistant"
        # If the chosen role collides with the tail AND flipping wouldn't
        # collide with the head, flip it.
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                # 两种角色都会冲突时，把 summary 合并进第一条尾部消息。
                _merge_summary_into_tail = True

        # summary 作为 user 消息时必须加结束标记，避免弱模型当成新请求。
        if not _merge_summary_into_tail and summary_role == "user":
            summary = (
                summary
                + "\n\n--- END OF CONTEXT SUMMARY — "
                "respond to the message below, not the summary above ---"
            )

        if not _merge_summary_into_tail:
            compressed.append({"role": summary_role, "content": summary})

        for i in range(compress_end, n_messages):
            msg = messages[i].copy()
            if _merge_summary_into_tail and i == compress_end:
                merged_prefix = (
                    summary
                    + "\n\n--- END OF CONTEXT SUMMARY — "
                    "respond to the message below, not the summary above ---\n\n"
                )
                msg["content"] = _append_text_to_content(
                    msg.get("content"),
                    merged_prefix,
                    prepend=True,
                )
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        # 历史图片统一替换成占位符，避免后续每轮重复发送大图片 payload。
        compressed = _strip_historical_media(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved_estimate = display_tokens - new_estimate

        # 记录压缩收益，低收益次数过多时 should_compress 会退避。
        savings_pct = (saved_estimate / display_tokens * 100) if display_tokens > 0 else 0
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved, %.0f%%)",
                n_messages,
                len(compressed),
                saved_estimate,
                savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed
