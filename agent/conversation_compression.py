"""上下文压缩编排层。

``ContextCompressor`` 只负责把消息列表压短；本模块负责把这件事接回
``AIAgent`` 运行时：检查辅助摘要模型是否可用、执行压缩、轮换 SQLite
session、重建 system prompt、通知 context engine / memory provider，并处理
图片过大时的重试缩图。

``run_agent.py`` 仍保留同名薄包装方法，旧调用点可以继续使用
``self._compress_context(...)``。
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from agent.model_metadata import estimate_request_tokens_rough

logger = logging.getLogger(__name__)


def _compression_lock_holder(agent: Any) -> str:
    """生成压缩锁持有者 ID，便于日志定位具体进程/线程/agent 实例。"""
    import threading
    return (
        f"pid={os.getpid()}"
        f":tid={threading.get_ident()}"
        f":agent={id(agent):x}"
        f":nonce={uuid.uuid4().hex[:8]}"
    )


def check_compression_model_feasibility(agent: Any) -> None:
    """检查辅助压缩模型上下文窗口是否足够，必要时降低本 session 压缩阈值。"""
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import (
            MINIMUM_CONTEXT_LENGTH,
            get_model_context_length,
        )

        client, aux_model = get_text_auxiliary_client(
            "compression",
            main_runtime=agent._current_main_runtime(),
        )
        # 尽量给辅助模型生成可读 provider 标签，方便用户定位配置问题。
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        if client is None or not aux_model:
            if _aux_cfg_provider and _aux_cfg_provider != "auto":
                msg = (
                    "⚠ Configured auxiliary compression provider "
                    f"'{_aux_cfg_provider}' is unavailable — context "
                    "compression will drop middle turns without a summary. "
                    "Check auxiliary.compression in config.yaml and "
                    "reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "No auxiliary LLM provider for compression — "
                "summaries will be unavailable."
            )
            return

        aux_base_url = str(getattr(client, "base_url", ""))
        # api_key 可能是可调用 bearer provider；解析上下文长度时不应为此签发 token。
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")

        aux_context = get_model_context_length(
            aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            # Each model must be resolved with its own provider so that
            # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
            # are invoked for the correct client, not inherited from the main model.
            provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")),
            custom_providers=agent._custom_providers,
        )

        # 辅助摘要模型也必须满足 Hermes 的最小上下文窗口要求。
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Auxiliary compression model {aux_model} has a context "
                f"window of {aux_context:,} tokens, which is below the "
                f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                f"Agent.  Choose a compression model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                f"auxiliary.compression.model in config.yaml), or set "
                f"auxiliary.compression.context_length to override the "
                f"detected value if it is wrong."
            )

        threshold = agent.context_compressor.threshold_tokens
        if aux_context < threshold:
            # 自动下调本 session 阈值，保证辅助模型能吃下待摘要窗口。
            old_threshold = threshold
            new_threshold = aux_context
            agent.context_compressor.threshold_tokens = new_threshold
            # 同步 threshold_percent，后续模型切换时能从新阈值比例重算。
            main_ctx = agent.context_compressor.context_length
            if main_ctx:
                agent.context_compressor.threshold_percent = (
                    new_threshold / main_ctx
                )
            safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
            # 生成 “model (provider)” 标签，让警告信息直接可读。
            _main_model = getattr(agent, "model", "") or "?"
            _main_provider = getattr(agent, "provider", "") or ""
            _aux_provider_label = (
                _aux_cfg_provider
                if _aux_cfg_provider and _aux_cfg_provider != "auto"
                else ""
            )
            if not _aux_provider_label:
                try:
                    from urllib.parse import urlparse
                    _aux_provider_label = (
                        urlparse(aux_base_url).hostname or aux_base_url
                    )
                except Exception:
                    _aux_provider_label = aux_base_url or "auto"
            _main_label = (
                f"{_main_model} ({_main_provider})"
                if _main_provider
                else _main_model
            )
            _aux_label = f"{aux_model} ({_aux_provider_label})"
            msg = (
                f"⚠ Compression model {_aux_label} context is "
                f"{aux_context:,} tokens, but the main model "
                f"{_main_label}'s compression threshold was "
                f"{old_threshold:,} tokens. "
                f"Auto-lowered this session's threshold to "
                f"{new_threshold:,} tokens so compression can run.\n"
                f"  To make this permanent, edit config.yaml — either:\n"
                f"  1. Use a larger compression model:\n"
                f"       auxiliary:\n"
                f"         compression:\n"
                f"           model: <model-with-{old_threshold:,}+-context>\n"
                f"  2. Lower the compression threshold:\n"
                f"       compression:\n"
                f"         threshold: 0.{safe_pct:02d}"
            )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression model %s has %d token context, "
                "below the main model's compression threshold of %d "
                "tokens — auto-lowered session threshold to %d to "
                "keep compression working.",
                aux_model,
                aux_context,
                old_threshold,
                new_threshold,
            )
    except ValueError:
        # 辅助模型低于硬性下限时必须阻止 session 启动。
        raise
    except Exception as exc:
        logger.debug(
            "Compression feasibility check failed (non-fatal): %s", exc
        )


def replay_compression_warning(agent: Any) -> None:
    """gateway 回调就绪后，补发初始化阶段缓存的压缩配置警告。"""
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        try:
            agent.status_callback("lifecycle", msg)
        except Exception:
            pass


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
) -> Tuple[list, str]:
    """执行压缩并在 SQLite 中切分/轮换 session，返回新消息列表和 system prompt。"""
    # 懒检查辅助压缩模型：只有真正要压缩时才做 provider/context lookup。
    if not getattr(agent, "_compression_feasibility_checked", False):
        # 检查成功后再置位；硬错误会向外抛出并阻止继续压缩。
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    _pre_msg_count = len(messages)
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
        agent.session_id or "none", _pre_msg_count,
        f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model,
        focus_topic,
    )
    agent._emit_status(
        "🗜️ Compacting context — summarizing earlier conversation so I can continue..."
    )

    # 压缩锁按旧 session_id 加锁，防止同一 session 的并发压缩产生孤儿子 session。
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    _lock_holder: Optional[str] = None
    # 锁子系统缺失时 fail-open：偶发 session fork 风险小于无限压缩空转。
    if _lock_db is not None and _lock_sid:
        _lock_holder = _compression_lock_holder(agent)
        try:
            _lock_acquired = _lock_db.try_acquire_compression_lock(
                _lock_sid, _lock_holder
            )
        except Exception as _lock_err:
            # 版本偏移或锁方法缺失时只告警一次，然后无锁继续。
            _lock_holder = None  # we don't own anything to release
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s "
                    "(%s: %s) — proceeding without lock. This usually means a "
                    "stale in-memory module after an update; restart the "
                    "process (or `hermes update`) to resync.",
                    _lock_sid, type(_lock_err).__name__, _lock_err,
                )
            _lock_acquired = True  # treat as acquired-but-unlocked; proceed
        if not _lock_acquired:
            try:
                existing = _lock_db.get_compression_lock_holder(_lock_sid)
            except Exception:
                existing = None
            logger.warning(
                "compression skipped: another path is compressing session=%s "
                "(holder=%s) — returning messages unchanged to avoid session fork",
                _lock_sid, existing,
            )
            _lock_holder = None  # don't release a lock we don't own
            # 面向用户只提示一次，避免 auto-compress 循环刷屏。
            if getattr(agent, "_last_compression_lock_warning_sid", None) != _lock_sid:
                agent._last_compression_lock_warning_sid = _lock_sid
                try:
                    agent._emit_warning(
                        "⚠ Skipping concurrent compression — another path "
                        "is already compressing this session. Will retry "
                        "after it finishes."
                    )
                except Exception:
                    pass
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp

    def _release_lock() -> None:
        """释放旧 session_id 上的压缩锁。"""
        if _lock_db is not None and _lock_sid and _lock_holder:
            try:
                _lock_db.release_compression_lock(_lock_sid, _lock_holder)
            except Exception as _rel_err:
                logger.debug("compression lock release failed: %s", _rel_err)

    # 压缩丢弃中间 turns 前，先通知外部 memory provider 抽取/同步。
    if agent._memory_manager:
        try:
            agent._memory_manager.on_pre_compress(messages)
        except Exception:
            pass

    try:
        compressed = agent.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic, force=force)
    except TypeError:
        # 兼容旧插件签名：不接受 focus_topic/force 时退回基础参数。
        compressed = agent.context_compressor.compress(messages, current_tokens=approx_tokens)
    except BaseException:
        # compress() 任意异常都必须释放锁，避免 session 永久卡住。
        _release_lock()
        raise

    # 压缩中止时不轮换 session，因为消息没有逻辑变化。
    if getattr(agent.context_compressor, "_last_compress_aborted", False):
        _err = getattr(agent.context_compressor, "_last_summary_error", None) or "unknown error"
        if getattr(agent, "_last_compression_summary_warning", None) != _err:
            agent._last_compression_summary_warning = _err
            agent._emit_warning(
                f"⚠ Compression aborted: {_err}. "
                "No messages were dropped — conversation continues unchanged. "
                "Run /compress to retry, or /new to start a fresh session."
            )
        _existing_sp = getattr(agent, "_cached_system_prompt", None)
        if not _existing_sp:
            _existing_sp = agent._build_system_prompt(system_message)
        _release_lock()  # compression aborted — no rotation will happen
        return messages, _existing_sp

    summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
    if summary_error:
        if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
            agent._last_compression_summary_warning = summary_error
            agent._emit_warning(
                f"⚠ Compression summary failed: {summary_error}. "
                "Inserted a fallback context marker."
            )
    else:
        # 即使压缩成功，也要暴露 auxiliary model 失败后回退主模型的事实。
        _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
        _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
        if _aux_fail_model:
            # 按 model/error 去重，避免每次压缩都重复提示。
            _aux_key = (_aux_fail_model, _aux_fail_err)
            if getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
                agent._last_aux_fallback_warning_key = _aux_key
                agent._emit_warning(
                    f"ℹ Configured compression model '{_aux_fail_model}' failed "
                    f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                    "check auxiliary.compression.model in config.yaml."
                )

    todo_snapshot = agent._todo_store.format_for_injection()
    if todo_snapshot:
        compressed.append({"role": "user", "content": todo_snapshot})

    agent._invalidate_system_prompt()
    new_system_prompt = agent._build_system_prompt(system_message)
    agent._cached_system_prompt = new_system_prompt

    if agent._session_db:
        try:
            # 压缩会生成子 session，标题按 lineage 自动编号继承。
            old_title = agent._session_db.get_session_title(agent.session_id)
            # 旧 session 结束前先提交 memory extraction。
            agent.commit_memory_session(messages)
            agent._session_db.end_session(agent.session_id, "compression")
            old_session_id = agent.session_id
            agent.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            # 先在 agent 线程更新 session ContextVar，gateway 稍后同步 SessionEntry。
            try:
                from gateway.session_context import set_current_session_id

                set_current_session_id(agent.session_id)
            except Exception:
                os.environ["HERMES_SESSION_ID"] = agent.session_id
            # 日志 session context 与工具/gateway context 分离，轮换后要单独同步。
            try:
                from hermes_logging import set_session_context

                set_session_context(agent.session_id)
            except Exception:
                pass
            agent._session_db_created = False
            agent._session_db.create_session(
                session_id=agent.session_id,
                source=agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                model=agent.model,
                model_config=agent._session_init_model_config,
                parent_session_id=old_session_id,
            )
            agent._session_db_created = True
            # Auto-number the title for the continuation session
            if old_title:
                try:
                    new_title = agent._session_db.get_next_title_in_lineage(old_title)
                    agent._session_db.set_session_title(agent.session_id, new_title)
                except (ValueError, Exception) as e:
                    logger.debug("Could not propagate title on compression: %s", e)
            agent._session_db.update_system_prompt(agent.session_id, new_system_prompt)
            # 新 session 尚未写入消息，flush 游标回到 0。
            agent._last_flushed_db_idx = 0
        except Exception as e:
            logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

    # 通知上下文引擎：这是 compression 触发的 session 轮换，不是全新会话。
    try:
        _old_sid = locals().get("old_session_id")
        if _old_sid and hasattr(agent.context_compressor, "on_session_start"):
            agent.context_compressor.on_session_start(
                agent.session_id or "",
                boundary_reason="compression",
                old_session_id=_old_sid,
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
    except Exception as _ce_err:
        logger.debug("context engine on_session_start (compression): %s", _ce_err)

    # 通知 memory provider session_id 已轮换，但逻辑对话仍在继续。
    try:
        _old_sid = locals().get("old_session_id")
        if _old_sid and agent._memory_manager:
            agent._memory_manager.on_session_switch(
                agent.session_id or "",
                parent_session_id=_old_sid,
                reset=False,
                reason="compression",
            )
    except Exception as _me_err:
        logger.debug("memory manager on_session_switch (compression): %s", _me_err)

    # 多次压缩会累积信息损失，提醒用户考虑 /new。
    _cc = agent.context_compressor.compression_count
    if _cc >= 2:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Session compressed {_cc} times — "
            f"accuracy may degrade. Consider /new to start fresh.",
            force=True,
        )

    # 压缩后只把粗略估算用于诊断，不当作真实 prompt usage。
    _compressed_est = estimate_request_tokens_rough(
        compressed,
        system_prompt=new_system_prompt or "",
        tools=agent.tools or None,
    )
    agent.context_compressor.last_compression_rough_tokens = _compressed_est
    agent.context_compressor.last_prompt_tokens = -1
    agent.context_compressor.last_completion_tokens = 0
    agent.context_compressor.awaiting_real_usage_after_compression = True

    # 压缩后清空文件读取去重，否则模型重读文件会拿到“未变化”占位而非全文。
    try:
        from tools.file_tools import reset_file_dedup
        reset_file_dedup(task_id)
    except Exception:
        pass

    logger.info(
        "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=true",
        agent.session_id or "none", _pre_msg_count, len(compressed),
        f"{_compressed_est:,}",
    )
    # 所有轮换收尾完成后再释放旧 session 锁，减少并发路径看到半完成状态。
    _release_lock()
    return compressed, new_system_prompt


def try_shrink_image_parts_in_messages(api_messages: list) -> bool:
    """Re-encode all native image parts at a smaller size to recover from
    image-too-large errors (Anthropic 5 MB, unknown other providers).

    Mutates ``api_messages`` in place. Returns True if any image part was
    actually replaced, False if there were no image parts to shrink or
    Pillow couldn't help (caller should surface the original error).

    Strategy: look for ``image_url`` / ``input_image`` parts carrying a
    ``data:image/...;base64,...`` payload.  For each one whose encoded
    size exceeds 4 MB (a safe target that slides under Anthropic's 5 MB
    ceiling with header overhead), write the base64 to a tempfile, call
    ``vision_tools._resize_image_for_vision`` to produce a smaller data
    URL, and substitute it in place.

    Non-data-URL images (http/https URLs) are not touched — the provider
    fetches those itself and the size limit is different.
    """
    if not api_messages:
        return False

    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False

    # 4 MB target leaves comfortable headroom under Anthropic's 5 MB.
    # Non-Anthropic providers we haven't observed rejecting are fine with
    # much larger; shrinking to 4 MB here loses quality but only fires
    # after a confirmed provider rejection, so the alternative is failure.
    target_bytes = 4 * 1024 * 1024
    # Anthropic enforces an 8000px per-side dimension cap independently of
    # the 5 MB byte cap.  A tall screenshot can be well under 5 MB yet far
    # over 8000px (e.g. 1200×12000 at 0.06 MB).  We check pixel dimensions
    # even when the byte budget is fine.
    max_dimension = 8000
    changed_count = 0
    # Track parts that are over the target but could NOT be shrunk under it.
    # If any survive, retrying is pointless — the same oversized payload will
    # be re-sent and rejected again, wasting the single retry budget.  We only
    # report success (caller retries) when every over-threshold image was
    # actually brought under the target.
    unshrinkable_oversized = 0

    def _shrink_data_url(url: str) -> Optional[str]:
        """Return a smaller data URL, or None if shrink can't help."""
        if not isinstance(url, str) or not url.startswith("data:"):
            return None

        # Check both byte size AND pixel dimensions.
        needs_shrink = len(url) > target_bytes  # over byte budget
        if not needs_shrink:
            # Even if bytes are fine, check pixel dimensions against
            # Anthropic's 8000px cap.  A tall image can be tiny in bytes
            # yet huge in pixels.
            try:
                import base64 as _b64_dim
                header_d, _, data_d = url.partition(",")
                if not data_d:
                    return None
                raw_d = _b64_dim.b64decode(data_d)
                from PIL import Image as _PILImage
                import io as _io_dim
                with _PILImage.open(_io_dim.BytesIO(raw_d)) as _img:
                    if max(_img.size) <= max_dimension:
                        return None  # both bytes and pixels are fine
                needs_shrink = True  # pixels exceed limit, force shrink
            except Exception:
                # If we can't check dimensions (Pillow unavailable, corrupt
                # image, etc.), fall back to byte-only check.
                return None

        try:
            header, _, data = url.partition(",")
            mime = "image/jpeg"
            if header.startswith("data:"):
                mime_part = header[len("data:"):].split(";", 1)[0].strip()
                if mime_part.startswith("image/"):
                    mime = mime_part
            import base64 as _b64
            raw = _b64.b64decode(data)
            suffix = {
                "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
            }.get(mime, ".jpg")
            tmp = tempfile.NamedTemporaryFile(
                prefix="hermes_shrink_", suffix=suffix, delete=False,
            )
            try:
                tmp.write(raw)
                tmp.close()
                resized = _resize_image_for_vision(
                    Path(tmp.name),
                    mime_type=mime,
                    max_base64_bytes=target_bytes,
                    max_dimension=max_dimension,
                )
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            if not resized or len(resized) >= len(url):
                # Shrink didn't help (or made it bigger — corrupt input?).
                return None
            return resized
        except Exception as exc:
            logger.warning("image-shrink recovery: re-encode failed — %s", exc)
            return None

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url")
            # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
            # OpenAI Responses: {"image_url": "data:..."}
            if isinstance(image_value, dict):
                url = image_value.get("url", "")
                resized = _shrink_data_url(url)
                if resized:
                    image_value["url"] = resized
                    changed_count += 1
                elif isinstance(url, str) and url.startswith("data:") \
                        and len(url) > target_bytes:
                    unshrinkable_oversized += 1
            elif isinstance(image_value, str):
                resized = _shrink_data_url(image_value)
                if resized:
                    part["image_url"] = resized
                    changed_count += 1
                elif image_value.startswith("data:") \
                        and len(image_value) > target_bytes:
                    unshrinkable_oversized += 1

    if changed_count:
        logger.info(
            "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
            changed_count, target_bytes / (1024 * 1024),
        )
    if unshrinkable_oversized:
        # At least one oversized image could not be shrunk under the target.
        # Retrying would re-send it and fail identically, so signal "no
        # progress" even if other parts shrank — the caller will surface the
        # original error rather than burning its single retry on a no-op.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)",
            unshrinkable_oversized, target_bytes / (1024 * 1024),
        )
        return False
    return changed_count > 0


__all__ = [
    "check_compression_model_feasibility",
    "replay_compression_warning",
    "compress_context",
    "try_shrink_image_parts_in_messages",
]
