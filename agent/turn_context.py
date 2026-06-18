"""每一轮 ``run_conversation`` 进入工具循环前的上下文准备。

这里集中处理一次用户输入进入主循环前的所有“前置上下文工程”：
stdio 防护、运行时主模型登记、重试计数重置、用户消息清洗、todo/memory
计数恢复、system prompt 复用或重建、崩溃恢复持久化、预检压缩、
``pre_llm_call`` 插件注入，以及外部记忆预取。

``build_turn_context`` 会有意修改 ``agent`` 的运行态；返回的
``TurnContext`` 只保存主循环后续要读取的局部变量。
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.iteration_budget import IterationBudget
from agent.model_metadata import estimate_request_tokens_rough

logger = logging.getLogger(__name__)


@dataclass
class TurnContext:
    """前置上下文阶段产物，供主工具循环继续使用。"""

    # 清洗后的用户消息，会进入本轮消息列表。
    user_message: str
    # 原始用户消息，用于 transcript / memory 查询，不带系统注入的 nudge。
    original_user_message: Any
    # 本轮工作消息列表，主循环会继续向其中追加 assistant/tool 消息。
    messages: List[Dict[str, Any]]
    # 预检压缩发生 session 轮换后会置空，避免旧历史被再次写入。
    conversation_history: Optional[List[Dict[str, Any]]]
    # 本轮使用的 system prompt；压缩后可能被重建。
    active_system_prompt: Optional[str]
    # 任务和轮次标识，贯穿工具调用、日志、状态回调。
    effective_task_id: str
    turn_id: str
    # 当前用户消息在 messages 中的下标。
    current_turn_user_idx: int
    # 本轮结束后是否触发 memory review。
    should_review_memory: bool = False
    # ``pre_llm_call`` 插件贡献的上下文，会追加到用户消息侧。
    plugin_user_context: str = ""
    # 外部记忆预取结果，主循环多次迭代复用。
    ext_prefetch_cache: str = ""


def build_turn_context(
    agent,
    user_message: str,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[str],
    *,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
) -> TurnContext:
    """执行单轮前置上下文准备，并返回主循环要消费的上下文对象。"""
    # 防止 headless/systemd 等环境里的断管错误中断会话。
    install_safe_stdio()

    agent._ensure_db_session()

    # 将本轮主模型运行时同步给 auxiliary_client，压缩/标题等辅助任务会用到。
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
        )
    except Exception:
        pass

    # 给当前线程日志打 session_id 标签，便于 hermes logs 过滤。
    set_session_context(agent.session_id)

    # 绑定 skill 写入来源，memory/skill 写操作会读取这个 ContextVar。
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # 如果上一轮启用了 fallback，这里恢复主运行时。
    agent._restore_primary_runtime()

    # 清理用户输入里的非法 surrogate 字符，避免 provider/JSON 编码失败。
    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # 保存流式回调，后续 _interruptible_api_call 会读取。
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    # 没有传入 task_id 时生成一个，隔离本轮工具 VM/缓存作用域。
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    turn_id = f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""

    # 每轮开始都重置工具/JSON/空响应等重试计数。
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    agent._vision_supported = True

    # 进入模型调用前清理上轮 provider 故障留下的死连接。
    if agent.api_mode != "anthropic_messages":
        try:
            if agent._cleanup_dead_connections():
                agent._emit_status(
                    "🔌 Detected stale connections from a previous provider "
                    "issue — cleaned up automatically. Proceeding with fresh "
                    "connection."
                )
        except Exception:
            pass
    # gateway 的 status_callback 初始化较晚，这里补发压缩配置警告。
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # memory/skill 的周期计数跨轮累计，不能在这里清零。
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # 记录本轮开始，用于调试和可观测性。
    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # 复制历史消息，避免原地修改调用方传入的列表。
    messages = list(conversation_history) if conversation_history else []

    # 从历史消息恢复 todo 状态。
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # 从历史消息恢复 memory nudge 计数，避免 resume 后节奏丢失。
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval

    # 统计用户轮次，供 memory flush 和周期提示使用。
    agent._user_turn_count += 1

    # 重置流式输出 scrubber，避免跨轮残留标签状态。
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # reasoning/think scrubber 也需要同样重置。
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # 保留用户原文，后续日志、memory 查询不能混入提示注入。
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # 判断本轮是否需要触发 memory review 提醒。
    should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    # 将用户消息加入本轮消息列表，这是后续上下文压缩和模型调用的基础。
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx

    if not agent.quiet_mode:
        _print_preview = summarize_user_message_for_log(user_message)
        agent._safe_print(
            f"💬 Starting conversation: '{_print_preview[:60]}"
            f"{'...' if len(_print_preview) > 60 else ''}'"
        )

    # System prompt 按 session 缓存，尽量保持 provider prefix cache 稳定。
    if agent._cached_system_prompt is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)

    active_system_prompt = agent._cached_system_prompt

    # 崩溃恢复：用户消息入队后尽早持久化。
    try:
        agent._persist_session(messages, conversation_history)
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )

    # 预检压缩：在真正 API 调用前，用粗略估算提前判断是否需要 compact。
    if (
        agent.compression_enabled
        and len(messages) > agent.context_compressor.protect_first_n
                            + agent.context_compressor.protect_last_n + 1
    ):
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )
        _compressor = agent.context_compressor
        _defer_preflight = getattr(
            _compressor,
            "should_defer_preflight_to_real_usage",
            lambda _tokens: False,
        )
        _preflight_deferred = _defer_preflight(_preflight_tokens)

        if not _preflight_deferred:
            _last = _compressor.last_prompt_tokens
            # Do NOT overwrite the -1 sentinel (#36718).
            if _last >= 0 and _preflight_tokens > _last:
                _compressor.last_prompt_tokens = _preflight_tokens

        if _preflight_deferred:
            logger.info(
                "Skipping preflight compression: rough estimate ~%s >= %s, "
                "but last real provider prompt was %s after compression",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                f"{_compressor.last_real_prompt_tokens:,}",
            )
        elif _compressor.should_compress(_preflight_tokens):
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            agent._emit_status(
                f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                f">= {_compressor.threshold_tokens:,} threshold. "
                "This may take a moment."
            )
            for _pass in range(3):
                _orig_len = len(messages)
                messages, active_system_prompt = agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if len(messages) >= _orig_len:
                    break  # 已无法进一步压缩，避免预检循环空转。
                conversation_history = None
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if not _compressor.should_compress(_preflight_tokens):
                    break

    # 插件可在 LLM 调用前追加上下文；注入到用户消息侧，不污染 system prompt。
    plugin_user_context = ""
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _pre_results = _invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=(not bool(conversation_history)),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
        _ctx_parts: list[str] = []
        for r in _pre_results:
            if isinstance(r, dict) and r.get("context"):
                _ctx_parts.append(str(r["context"]))
            elif isinstance(r, str) and r.strip():
                _ctx_parts.append(r)
        if _ctx_parts:
            plugin_user_context = "\n\n".join(_ctx_parts)
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)

    # 本轮文件修改校验状态。
    agent._turn_failed_file_mutations = {}

    # 记录执行线程，确保 interrupt 只影响当前 agent 线程。
    agent._execution_thread_id = threading.current_thread().ident

    # 清掉旧线程中断状态，同时保留已经发起的 pending interrupt。
    ra()._set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        ra()._set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False

    # 在 memory 预取前通知 provider 新一轮开始。
    if agent._memory_manager:
        try:
            _turn_msg = original_user_message if isinstance(original_user_message, str) else ""
            agent._memory_manager.on_turn_start(agent._user_turn_count, _turn_msg)
        except Exception:
            pass

    # 外部记忆每轮只预取一次，主循环内部复用。
    ext_prefetch_cache = ""
    if agent._memory_manager:
        try:
            _query = original_user_message if isinstance(original_user_message, str) else ""
            ext_prefetch_cache = agent._memory_manager.prefetch_all(_query) or ""
        except Exception:
            pass

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context,
        ext_prefetch_cache=ext_prefetch_cache,
    )
