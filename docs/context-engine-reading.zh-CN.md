# Hermes Agent 上下文工程阅读文档

这份文档用于在 GitHub 上按调用顺序阅读 Hermes 的上下文工程代码。链接均为仓库内相对链接，适合直接在 GitHub README / Docs 页面里跳转。

Hermes 的上下文工程不是一个单独文件，而是一条贯穿 **初始化、每轮对话、模型调用、工具调用、压缩、session 轮换、插件扩展** 的链路。核心目标是：在模型上下文窗口有限、工具输出可能很大、会话可能很长的情况下，让模型每轮都看到足够新、足够准、足够安全的上下文。

## GitHub 跳转索引

- [AIAgent.__init__](../run_agent.py#L343)
- [agent_init 中选择上下文引擎](../agent/agent_init.py#L1465)
- [ContextEngine 接口](../agent/context_engine.py#L23)
- [ContextCompressor 默认实现](../agent/context_compressor.py#L422)
- [AIAgent.run_conversation](../run_agent.py#L5092)
- [conversation_loop.run_conversation](../agent/conversation_loop.py#L371)
- [build_turn_context](../agent/turn_context.py#L53)
- [compress_context 压缩编排](../agent/conversation_compression.py#L195)
- [ContextCompressor.compress 默认压缩入口](../agent/context_compressor.py#L1531)
- [context_references 显式 @ 引用](../agent/context_references.py#L66)
- [plugins/context_engine 插件引擎发现](../plugins/context_engine/__init__.py#L23)

## 1. 初始化和上下文引擎选择

调用顺序：

1. [AIAgent.__init__](../run_agent.py#L343)
2. [agent_init 中读取 context.engine 并选择引擎](../agent/agent_init.py#L1465)
3. [load_context_engine](../plugins/context_engine/__init__.py#L64)
4. [_load_engine_from_dir](../plugins/context_engine/__init__.py#L82)
5. [_EngineCollector.register_context_engine](../plugins/context_engine/__init__.py#L183)
6. [get_plugin_context_engine](../hermes_cli/plugins.py#L1820)
7. [ContextCompressor.__init__](../agent/context_compressor.py#L479)
8. [get_model_context_length](../agent/model_metadata.py#L1501)
9. [ContextEngine.update_model](../agent/context_engine.py#L126) 或 [ContextCompressor.update_model](../agent/context_compressor.py#L452)
10. [ContextEngine.get_tool_schemas](../agent/context_engine.py#L100)

原理：

Hermes 初始化时会读取配置里的 `context.engine`。如果值是 `compressor`，就使用内置 [ContextCompressor](../agent/context_compressor.py#L422)。如果是其他名字，会先查仓库内的 `plugins/context_engine/<name>/`，再查通用插件系统已经注册的上下文引擎。找不到时回退到内置压缩器。

这个设计把“上下文治理策略”从主循环里拆出来。主循环不需要知道压缩器内部是摘要、DAG、向量检索还是别的结构；它只调用统一的 [ContextEngine](../agent/context_engine.py#L23) 接口。

读代码时先看 [agent_init 中选择上下文引擎](../agent/agent_init.py#L1465)。这里还能看到两个关键约束：一是引擎必须知道当前模型的 context window，二是低于 Hermes 最小上下文窗口的模型会被拒绝启动，因为工具工作流需要足够大的输入窗口。

## 2. ContextEngine 抽象接口

调用顺序：

1. [ContextEngine.name](../agent/context_engine.py#L27)
2. [ContextEngine.update_from_response](../agent/context_engine.py#L51)
3. [ContextEngine.should_compress](../agent/context_engine.py#L55)
4. [ContextEngine.compress](../agent/context_engine.py#L59)
5. [ContextEngine.should_compress_preflight](../agent/context_engine.py#L69)
6. [ContextEngine.should_defer_preflight_to_real_usage](../agent/context_engine.py#L73)
7. [ContextEngine.has_content_to_compress](../agent/context_engine.py#L79)
8. [ContextEngine.on_session_start](../agent/context_engine.py#L85)
9. [ContextEngine.on_session_end](../agent/context_engine.py#L88)
10. [ContextEngine.on_session_reset](../agent/context_engine.py#L91)
11. [ContextEngine.get_tool_schemas](../agent/context_engine.py#L100)
12. [ContextEngine.handle_tool_call](../agent/context_engine.py#L104)
13. [ContextEngine.get_status](../agent/context_engine.py#L111)
14. [ContextEngine.update_model](../agent/context_engine.py#L126)

原理：

[ContextEngine](../agent/context_engine.py#L23) 是主循环和上下文实现之间的契约。主循环只问四类问题：模型刚返回了多少 token、现在是否该压缩、如果压缩应该返回什么消息列表、session 生命周期变化时引擎是否需要同步状态。

虽然很多字段还叫 `context_compressor`，但现在语义更宽：它代表“当前激活的上下文引擎”。默认实现确实是压缩器，插件实现也会挂在这个属性上。

## 3. 入口和每轮主循环

调用顺序：

1. [AIAgent.run_conversation](../run_agent.py#L5092)
2. [conversation_loop.run_conversation](../agent/conversation_loop.py#L371)
3. [build_turn_context](../agent/turn_context.py#L53)
4. [_restore_or_build_system_prompt](../agent/conversation_loop.py#L225)
5. [estimate_request_tokens_rough](../agent/model_metadata.py#L1924)
6. [ContextEngine.should_defer_preflight_to_real_usage](../agent/context_engine.py#L73)
7. [ContextEngine.should_compress](../agent/context_engine.py#L55)
8. [AIAgent._compress_context](../run_agent.py#L4938)
9. [compress_context](../agent/conversation_compression.py#L195)
10. [normalize_usage](../agent/usage_pricing.py#L700)
11. [ContextEngine.update_from_response](../agent/context_engine.py#L51)
12. [AIAgent._execute_tool_calls](../run_agent.py#L4991)

原理：

用户输入不是简单 append 到历史消息后就发给模型。Hermes 会先进入 [build_turn_context](../agent/turn_context.py#L53)，完成每轮准备：清洗输入、恢复模型状态、构建或复用 system prompt、尽早持久化用户消息、做压缩预检、注入插件上下文、预取 memory。

预检压缩发生在真正 API 调用之前，此时还没有 provider 返回的真实 usage，只能走 [estimate_request_tokens_rough](../agent/model_metadata.py#L1924)。如果估算结果已经接近阈值，会通过 [AIAgent._compress_context](../run_agent.py#L4938) 进入压缩编排。

模型响应回来后，主循环再用 [normalize_usage](../agent/usage_pricing.py#L700) 把不同 provider 的 usage 统一成标准字段，并调用 [ContextEngine.update_from_response](../agent/context_engine.py#L51)。这一步比预检更可信，因为它来自 provider 的真实计数。

## 4. 每轮上下文准备 build_turn_context

调用顺序：

1. [conversation_loop.run_conversation](../agent/conversation_loop.py#L371)
2. [build_turn_context](../agent/turn_context.py#L53)
3. [_restore_or_build_system_prompt](../agent/conversation_loop.py#L225)
4. [AIAgent._build_system_prompt](../run_agent.py#L3154)
5. [estimate_request_tokens_rough](../agent/model_metadata.py#L1924)
6. [ContextEngine.should_defer_preflight_to_real_usage](../agent/context_engine.py#L73)
7. [ContextEngine.should_compress](../agent/context_engine.py#L55)
8. [AIAgent._compress_context](../run_agent.py#L4938)
9. [invoke_hook](../hermes_cli/plugins.py#L1715)

原理：

[build_turn_context](../agent/turn_context.py#L53) 是每轮对话的 prologue。它会复制历史消息，追加当前用户输入，并记录当前 user message 的下标。这个下标后续会被 session DB、memory review、工具调用持久化等逻辑使用。

system prompt 会按 session 缓存。只要没有压缩、模型切换或配置变化，Hermes 会尽量复用它，以提升 provider prefix cache 命中率。压缩后必须调用 [AIAgent._invalidate_system_prompt](../run_agent.py#L3307) 和 [AIAgent._build_system_prompt](../run_agent.py#L3154) 重建。

插件上下文通过 [invoke_hook](../hermes_cli/plugins.py#L1715) 进入本轮用户侧上下文，而不是直接污染 system prompt。这样插件可以提供“本轮相关信息”，同时不破坏 system prompt 的稳定性。

## 5. 真实 usage 和压缩触发

调用顺序：

1. [conversation_loop.run_conversation](../agent/conversation_loop.py#L371)
2. [normalize_usage](../agent/usage_pricing.py#L700)
3. [ContextCompressor.update_from_response](../agent/context_compressor.py#L565)
4. [ContextCompressor.should_compress](../agent/context_compressor.py#L600)
5. [AIAgent._compress_context](../run_agent.py#L4938)
6. [compress_context](../agent/conversation_compression.py#L195)

原理：

压缩判断优先看 `prompt_tokens`，而不是把 `completion_tokens` 一起当作主要压力来源。原因是下一轮请求是否能塞进 context window，主要取决于输入侧 prompt。thinking/reasoning 模型可能把大量 reasoning 记到输出侧，如果过度依赖 completion tokens，会过早压缩。

[ContextCompressor.should_compress](../agent/context_compressor.py#L600) 还有抗空转逻辑：如果最近几次压缩节省很少，它会暂时跳过，避免每轮都压缩但上下文没有实际变短。

## 6. 压缩编排层 compress_context

调用顺序：

1. [AIAgent._compress_context](../run_agent.py#L4938)
2. [compress_context](../agent/conversation_compression.py#L195)
3. [check_compression_model_feasibility](../agent/conversation_compression.py#L38)
4. [MemoryManager.on_pre_compress](../agent/memory_manager.py#L686)
5. [ContextEngine.compress](../agent/context_engine.py#L59)
6. [ContextCompressor.compress](../agent/context_compressor.py#L1531)
7. [AIAgent._invalidate_system_prompt](../run_agent.py#L3307)
8. [AIAgent._build_system_prompt](../run_agent.py#L3154)
9. [AIAgent.commit_memory_session](../run_agent.py#L2910)
10. [ContextEngine.on_session_start](../agent/context_engine.py#L85)
11. [MemoryManager.on_session_switch](../agent/memory_manager.py#L638)
12. [reset_file_dedup](../tools/file_tools.py#L964)

原理：

[ContextCompressor.compress](../agent/context_compressor.py#L1531) 只负责把消息列表变短，但一次真实压缩还会影响 session DB、memory、system prompt、文件读取缓存和插件状态，所以外围有 [compress_context](../agent/conversation_compression.py#L195)。

压缩前会先检查 auxiliary compression 模型是否可用，并确认它的 context window 足够容纳摘要输入。然后会按旧 session_id 获取压缩锁，避免主 agent 和后台路径同时压缩同一段历史，导致 session fork。

真正丢弃中间 turns 前，会先调用 [MemoryManager.on_pre_compress](../agent/memory_manager.py#L686)，给 memory provider 一个同步关键信息的机会。压缩成功后会重建 system prompt、结束旧 session、创建新 session，并调用 [MemoryManager.on_session_switch](../agent/memory_manager.py#L638) 让 memory provider 跟随 session 轮换。

最后调用 [reset_file_dedup](../tools/file_tools.py#L964) 很关键：压缩后旧文件全文可能只剩摘要，如果模型再次读取同一个文件，应该拿到全文，而不是“文件未变化”的占位。

## 7. 默认压缩器内部算法

调用顺序：

1. [ContextCompressor.compress](../agent/context_compressor.py#L1531)
2. [ContextCompressor._prune_old_tool_results](../agent/context_compressor.py#L621)
3. [ContextCompressor._protect_head_size](../agent/context_compressor.py#L1390)
4. [ContextCompressor._align_boundary_forward](../agent/context_compressor.py#L1384)
5. [ContextCompressor._find_tail_cut_by_tokens](../agent/context_compressor.py#L1450)
6. [ContextCompressor._align_boundary_backward](../agent/context_compressor.py#L1397)
7. [ContextCompressor._ensure_last_user_message_in_tail](../agent/context_compressor.py#L1423)
8. [ContextCompressor._find_latest_context_summary](../agent/context_compressor.py#L1312)
9. [ContextCompressor._generate_summary](../agent/context_compressor.py#L1012)
10. [ContextCompressor._compute_summary_budget](../agent/context_compressor.py#L757)
11. [ContextCompressor._serialize_for_summary](../agent/context_compressor.py#L770)
12. [ContextCompressor._build_static_fallback_summary](../agent/context_compressor.py#L816)
13. [ContextCompressor._with_summary_prefix](../agent/context_compressor.py#L1299)
14. [ContextCompressor._sanitize_tool_pairs](../agent/context_compressor.py#L1336)
15. [_strip_historical_media](../agent/context_compressor.py#L270)

原理：

默认压缩器不是简单把前半段聊天总结一下，而是把消息拆成三段：

```text
head（受保护开头） + middle（要摘要的旧消息） + tail（受保护最近上下文）
```

head 通常保留 system prompt 和配置指定的开头消息；tail 按 token 预算保留，而不是只按消息数量保留。这样最近一次用户任务、刚执行的工具结果、最新错误信息更容易留在原文里。

[_prune_old_tool_results](../agent/context_compressor.py#L621) 会先本地裁剪旧工具输出。这一步不消耗 LLM 调用，可以先把特别大的 stdout、文件内容、搜索结果压成较短摘要。随后 [_find_tail_cut_by_tokens](../agent/context_compressor.py#L1450) 从尾部往前累计 token，决定最近上下文从哪里开始受保护。

边界对齐很重要。[_align_boundary_forward](../agent/context_compressor.py#L1384) 和 [_align_boundary_backward](../agent/context_compressor.py#L1397) 避免切开 `assistant tool_call -> tool result` 这一组消息。压缩后还会调用 [_sanitize_tool_pairs](../agent/context_compressor.py#L1336)，删除孤儿 tool result，或给保留下来的 tool_call 补占位结果，保证 provider 接受消息序列。

摘要生成走 [_generate_summary](../agent/context_compressor.py#L1012)。它会把待压缩 turns 通过 [_serialize_for_summary](../agent/context_compressor.py#L770) 序列化，并按 [_compute_summary_budget](../agent/context_compressor.py#L757) 分配摘要 token。如果 LLM 摘要失败，则视配置决定中止压缩，或用 [_build_static_fallback_summary](../agent/context_compressor.py#L816) 生成本地兜底摘要。

## 8. tool_call / tool_result 完整性

调用顺序：

1. [ContextCompressor._get_tool_call_id](../agent/context_compressor.py#L1330)
2. [ContextCompressor._sanitize_tool_pairs](../agent/context_compressor.py#L1336)
3. [ContextCompressor._align_boundary_forward](../agent/context_compressor.py#L1384)
4. [ContextCompressor._align_boundary_backward](../agent/context_compressor.py#L1397)

原理：

OpenAI 格式要求 assistant 里出现的每个 `tool_call_id` 都要有对应的 `role="tool"` 结果。压缩时如果只保留其中一半，下一次 provider 调用可能直接报错。

Hermes 用边界对齐减少“切半个工具组”的概率，再用 [_sanitize_tool_pairs](../agent/context_compressor.py#L1336) 做最后修复：孤儿 tool result 会被删除，缺结果的 tool_call 会补一个占位 tool result，提示模型去看上方摘要。

## 9. 显式 @ 引用上下文

调用顺序：

1. [preprocess_context_references](../agent/context_references.py#L110)
2. [preprocess_context_references_async](../agent/context_references.py#L138)
3. [parse_context_references](../agent/context_references.py#L66)
4. [_expand_reference](../agent/context_references.py#L212)
5. [_resolve_path](../agent/context_references.py#L341)
6. [_ensure_reference_path_allowed](../agent/context_references.py#L355)
7. [_fetch_url_content](../agent/context_references.py#L315)
8. [_default_url_fetcher](../agent/context_references.py#L328)
9. [_remove_reference_tokens](../agent/context_references.py#L424)

原理：

`@file`、`@folder`、`@git`、`@url` 这类显式引用会在模型调用前展开成“Attached Context”。它属于输入增强，不属于长会话压缩。

[parse_context_references](../agent/context_references.py#L66) 负责从用户消息里识别引用 token，[preprocess_context_references_async](../agent/context_references.py#L138) 负责异步展开。文件路径会经过 [_resolve_path](../agent/context_references.py#L341) 和 [_ensure_reference_path_allowed](../agent/context_references.py#L355) 保护，避免越权读取。展开后 [_remove_reference_tokens](../agent/context_references.py#L424) 会把原始 `@...` token 从用户可见文本里移掉，再追加实际上下文块。

## 10. 上下文引擎专属工具

调用顺序：

1. [ContextEngine.get_tool_schemas](../agent/context_engine.py#L100)
2. [agent_init 注入上下文引擎工具 schema](../agent/agent_init.py#L1530)
3. [AIAgent._execute_tool_calls](../run_agent.py#L4991)
4. [tool_executor 中转发给 context_compressor.handle_tool_call](../agent/tool_executor.py#L1118)
5. [ContextEngine.handle_tool_call](../agent/context_engine.py#L104)

原理：

插件上下文引擎可以暴露自己的工具。例如一个更高级的上下文引擎可能提供“检索压缩图谱”“展开历史节点”“搜索长期上下文”的工具。初始化时 [ContextEngine.get_tool_schemas](../agent/context_engine.py#L100) 返回 schema，agent_init 会把它注入可用工具列表。

工具执行时，普通工具走常规 tool executor；如果工具名属于上下文引擎工具集合，就转发给 [ContextEngine.handle_tool_call](../agent/context_engine.py#L104)。这样上下文引擎可以拥有自己的交互式检索能力，但仍复用主 agent 的工具调用协议。

## 11. 图片过大恢复

调用顺序：

1. [try_shrink_image_parts_in_messages](../agent/conversation_compression.py#L454)
2. [tools.vision_tools._resize_image_for_vision](../tools/vision_tools.py#L372)

原理：

多模态消息里如果携带 data URL 图片，provider 可能因为字节数或像素尺寸拒绝请求。[try_shrink_image_parts_in_messages](../agent/conversation_compression.py#L454) 是错误恢复路径：它会原地找到 `image_url` / `input_image` 中的 base64 data URL，调用 vision 工具缩图，再重试请求。

这和普通上下文压缩不是一回事。普通压缩处理长历史；图片缩小处理“当前 API payload 被 provider 拒绝”的恢复问题。

## 12. 推荐阅读顺序

第一次阅读建议按这个顺序：

1. [ContextEngine](../agent/context_engine.py#L23)：先理解主循环和上下文引擎的接口边界。
2. [agent_init 中选择上下文引擎](../agent/agent_init.py#L1465)：看默认压缩器和插件引擎如何接入。
3. [conversation_loop.run_conversation](../agent/conversation_loop.py#L371)：看每轮主循环如何进入上下文准备、模型调用、工具调用。
4. [build_turn_context](../agent/turn_context.py#L53)：看模型调用前的输入增强和预检压缩。
5. [compress_context](../agent/conversation_compression.py#L195)：看压缩如何影响 session、memory、system prompt。
6. [ContextCompressor.compress](../agent/context_compressor.py#L1531)：看默认压缩算法的真实切分、摘要和修复逻辑。
7. [context_references](../agent/context_references.py#L66)：最后看显式 `@` 引用如何把文件/URL/git 内容注入本轮上下文。

读完这条链路后，再看 gateway、TUI 或 desktop 时会清楚很多：那些入口最终都要落回同一套 `run_conversation -> build_turn_context -> provider/tool loop -> compress_context` 的上下文治理模型。
