# Hermes Agent 上下文工程阅读文档

这份文档是为了“先理解原理，再跳进代码”。Hermes 的上下文工程不是单一模块，而是一条贯穿初始化、每轮对话、模型调用、工具调用、长会话压缩、session 存储和插件扩展的链路。

核心目标只有一个：**在模型上下文窗口有限、工具输出可能很大、会话可能很长的情况下，让模型每轮都看到“足够新、足够准、足够安全”的上下文。**

## 总体心智模型

一次普通用户输入进入 Hermes 后，大致经历三段：

```text
用户输入
  -> 每轮上下文准备
     -> 组装历史消息、system prompt、插件上下文、memory 预取、显式 @ 引用
  -> 模型/工具循环
     -> 调模型、处理工具调用、记录真实 token usage
  -> 上下文治理
     -> 判断是否压缩、执行摘要、修复工具消息、轮换 session
```

可以把它理解成三套机制互相配合：

1. **输入增强**：让模型在本轮看到必要的文件、diff、URL、memory、插件上下文。
2. **窗口管理**：用真实 token usage 和粗略估算判断是否接近 context window。
3. **压缩续航**：把旧的中间对话变成结构化 handoff summary，让会话继续跑。

## 先看哪些代码

建议按这个顺序看源码：

GitHub 跳转索引：

- [ContextEngine 接口](../agent/context_engine.py#L23)
- [上下文引擎初始化选择](../agent/agent_init.py#L1465)
- [每轮上下文准备 build_turn_context](../agent/turn_context.py#L53)
- [主循环 turn setup](../agent/conversation_loop.py#L400)
- [主循环 usage 更新](../agent/conversation_loop.py#L1578)
- [压缩编排 compress_context](../agent/conversation_compression.py#L196)
- [默认压缩器 ContextCompressor](../agent/context_compressor.py#L430)
- [工具输出裁剪](../agent/context_compressor.py#L644)
- [结构化摘要生成](../agent/context_compressor.py#L1041)
- [tool_call/tool_result 修复](../agent/context_compressor.py#L1391)
- [head 保护计算](../agent/context_compressor.py#L1445)
- [tail token 预算切点](../agent/context_compressor.py#L1505)
- [默认压缩入口](../agent/context_compressor.py#L1586)
- [`@` 引用解析](../agent/context_references.py#L66)
- [`@` 引用展开](../agent/context_references.py#L138)
- [内置上下文引擎插件发现](../plugins/context_engine/__init__.py#L23)

1. [agent/context_engine.py](../agent/context_engine.py#L23)
   - 先看 [ContextEngine](../agent/context_engine.py#L23) 抽象接口。
   - 重点方法：[update_from_response](../agent/context_engine.py#L51)、[should_compress](../agent/context_engine.py#L55)、[compress](../agent/context_engine.py#L59)、[on_session_start](../agent/context_engine.py#L83)、[get_tool_schemas](../agent/context_engine.py#L101)。

2. [agent/agent_init.py](../agent/agent_init.py#L1465)
   - 看“选择上下文引擎”的初始化逻辑。
   - 这里决定使用内置压缩器，还是 `context.engine` 指定的插件引擎。

3. [agent/turn_context.py](../agent/turn_context.py#L1)
   - 看 [TurnContext](../agent/turn_context.py#L27) 和 [build_turn_context](../agent/turn_context.py#L53)。
   - 这里是每轮用户输入进入模型前的上下文准备。

4. [agent/conversation_loop.py](../agent/conversation_loop.py#L400)
   - 看主循环如何调用 `build_turn_context`。
   - 再看模型响应后如何调用 [update_from_response](../agent/conversation_loop.py#L1578) 和压缩判断。

5. [agent/conversation_compression.py](../agent/conversation_compression.py#L196)
   - 看 [compress_context](../agent/conversation_compression.py#L196)。
   - 这是“压缩器结果”接回 session DB、memory、日志和 system prompt 的编排层。

6. [agent/context_compressor.py](../agent/context_compressor.py#L430)
   - 看默认实现 [ContextCompressor](../agent/context_compressor.py#L430)。
   - 重点方法：[compress](../agent/context_compressor.py#L1586)、[_generate_summary](../agent/context_compressor.py#L1041)、[_prune_old_tool_results](../agent/context_compressor.py#L644)、[_find_tail_cut_by_tokens](../agent/context_compressor.py#L1505)、[_sanitize_tool_pairs](../agent/context_compressor.py#L1391)。

7. [agent/context_references.py](../agent/context_references.py#L66)
   - 看显式 `@` 引用解析：[parse_context_references](../agent/context_references.py#L66)、[preprocess_context_references_async](../agent/context_references.py#L138)。

8. [plugins/context_engine/__init__.py](../plugins/context_engine/__init__.py#L23)
   - 看内置上下文引擎插件发现：[discover_context_engines](../plugins/context_engine/__init__.py#L23)、[load_context_engine](../plugins/context_engine/__init__.py#L64)、[_EngineCollector](../plugins/context_engine/__init__.py#L175)。

## 1. 上下文引擎接口

入口：[ContextEngine](../agent/context_engine.py#L23)

Hermes 把“上下文怎么治理”抽象成一个 `ContextEngine`。默认实现是 `ContextCompressor`，但插件可以替换它。

接口设计的核心思想是：**主循环只关心几个标准动作，不关心压缩器内部是摘要、DAG、向量检索还是别的结构。**

主循环需要问上下文引擎几个问题：

- 模型刚返回了多少 token？调用 [update_from_response](../agent/context_engine.py#L51) 更新状态。
- 现在该压缩吗？调用 [should_compress](../agent/context_engine.py#L55)。
- 如果要压缩，给你完整消息列表，你返回新消息列表。调用 [compress](../agent/context_engine.py#L59)。
- session 开始/结束了，你要不要加载或释放状态？调用 [on_session_start](../agent/context_engine.py#L83) / `on_session_end`。
- 你有没有自己的工具要暴露给模型？调用 [get_tool_schemas](../agent/context_engine.py#L101)。

这也是为什么代码里虽然属性名还叫 `context_compressor`，但实际含义已经更宽：它是“当前激活的上下文引擎”。这个命名是历史遗留，读代码时要脑内翻译一下。

## 2. 初始化时怎么选择上下文引擎

入口：[agent_init.py 选择上下文引擎](../agent/agent_init.py#L1465)

初始化阶段会读取配置：

```yaml
context:
  engine: compressor
```

选择逻辑：

```text
context.engine == "compressor"
  -> 使用内置 ContextCompressor

context.engine != "compressor"
  -> 先查 plugins/context_engine/<name>/
  -> 再查通用插件系统注册的 context engine
  -> 都找不到则回退 ContextCompressor
```

初始化还会做三件重要的事：

1. **解析当前模型上下文长度**
   - 插件引擎和默认压缩器都需要知道 `context_length`。
   - 因为不同 provider 对同名模型可能有不同上下文窗口。

2. **检查最小上下文窗口**
   - Hermes 工具工作流要求模型上下文不能太小。
   - 低于 `MINIMUM_CONTEXT_LENGTH` 会拒绝启动。

3. **注入上下文引擎工具**
   - 如果引擎实现了 `get_tool_schemas()`，初始化会把这些 schema 加到 `agent.tools`。
   - 例如某些引擎可能暴露 `lcm_grep`、`lcm_expand` 之类工具。
   - 这些工具受 `enabled_toolsets` 的 `context_engine` 开关控制。

## 3. 每轮上下文准备

入口：[build_turn_context](../agent/turn_context.py#L53)

`conversation_loop` 真正开始调用模型前，会先执行一段“turn prologue”。这段逻辑被抽到 `build_turn_context()`。

它解决的问题是：**一轮用户消息不是简单 append 到 history 就完事了。模型调用前要恢复很多运行态。**

主要步骤：

1. **基础运行态恢复**
   - 设置当前 session 的日志上下文。
   - 恢复主 provider/model，避免上一轮 fallback 泄漏到本轮。
   - 清理 surrogate 字符，避免 JSON/provider 编码错误。
   - 重置工具调用、JSON、空响应等重试计数。

2. **消息列表准备**
   - 复制 `conversation_history`，避免修改调用方传入的列表。
   - 把当前用户消息 append 到 `messages`。
   - 记录当前 user message 的下标，后续持久化和 memory 逻辑会用。

3. **system prompt 缓存**
   - 如果当前 session 已缓存 system prompt，就复用。
   - 这样可以提升 provider prefix cache 命中率。
   - 如果压缩发生，system prompt 会被重建。

4. **崩溃恢复持久化**
   - 用户消息刚进入消息列表，就尽早写 session DB。
   - 这样即使后续工具或 provider 崩溃，也能恢复用户这轮输入。

5. **预检压缩**
   - 还没真正调用 provider，所以没有真实 token usage。
   - 只能用 `estimate_request_tokens_rough(...)` 估算。
   - 如果粗略估算已经超过阈值，会先调用 `agent._compress_context(...)`。

6. **插件上下文注入**
   - 调用 `pre_llm_call` hook。
   - 插件返回的上下文会拼到用户消息侧，而不是污染 system prompt。

7. **memory 预取**
   - 外部 memory provider 每轮预取一次。
   - 主循环多次工具迭代时复用这个预取结果。

## 4. 主循环里怎么触发压缩

入口：[conversation_loop.py turn setup](../agent/conversation_loop.py#L400)，[usage 更新](../agent/conversation_loop.py#L1578)

模型响应后，Hermes 会读取 provider 返回的 usage：

```text
response.usage
  -> normalize_usage(...)
  -> usage_dict
  -> context_compressor.update_from_response(usage_dict)
```

然后用上下文引擎判断是否需要压缩：

```text
if compression_enabled and context_compressor.should_compress(tokens):
    messages, active_system_prompt = agent._compress_context(...)
```

这里有一个关键原则：**压缩判断主要看 prompt tokens，而不是 completion tokens。**

原因：

- 下一轮请求能不能塞进 context window，主要取决于输入侧 prompt。
- thinking/reasoning 模型可能把大量 reasoning 记到 completion tokens。
- 如果把 completion/reasoning 也算进压缩压力，会过早压缩。

如果 provider 没返回 usage，代码会退回粗略估算。粗略估算会把工具 schema 也算进去，因为几十个工具 schema 可能占 20K 到 30K token，忽略它会导致压缩触发太晚。

## 5. 压缩编排层做什么

入口：[compress_context](../agent/conversation_compression.py#L196)

`ContextCompressor.compress()` 只负责“把 messages 变短”。但一次真实压缩还牵涉很多外围状态，所以有 `conversation_compression.py` 这一层。

它负责：

1. **检查 auxiliary compression 模型**
   - 压缩摘要通常用辅助模型生成。
   - 如果辅助模型上下文窗口小于主模型压缩阈值，摘要请求本身可能塞不进去。
   - 代码会必要时降低本 session 的压缩阈值。

2. **获取压缩锁**
   - 同一个 session 可能被主 agent 和后台 review agent 同时看到。
   - 如果两个路径同时压缩同一段历史，可能创建两个子 session，其中一个变成孤儿。
   - 所以压缩锁按旧 session_id 加锁。

3. **通知 memory provider**
   - 压缩会丢掉中间 turns 的原文。
   - 丢之前先让 memory provider 抽取或同步关键信息。

4. **调用上下文引擎压缩**
   - 默认调用 [ContextCompressor.compress](../agent/context_compressor.py#L1586)。
   - 插件引擎也走同一个接口。

5. **重建 system prompt**
   - 压缩后 active messages 变化，system prompt 缓存需要失效并重建。

6. **轮换 SQLite session**
   - 旧 session 以 `compression` 结束。
   - 新 session 继承 parent_session_id 和标题 lineage。
   - 这样历史检索和 UI 上能看到压缩边界。

7. **通知 context engine / memory provider session 已切换**
   - 对话逻辑继续，但 session_id 和 DB row 变了。
   - 插件引擎可能要继承 DAG lineage。
   - memory provider 可能要刷新 per-session 缓存。

8. **清理文件读取去重缓存**
   - 压缩后旧文件全文可能只剩摘要。
   - 如果模型再次读取同一个文件，需要拿到全文，而不是“文件未变化”的占位。

## 6. 默认压缩器的核心算法

入口：[ContextCompressor.compress](../agent/context_compressor.py#L1586)

默认算法不是简单“把前面聊天总结一下”。它更像是一次有约束的消息重写：

```text
原始消息：
  head + old middle + recent tail

压缩后：
  head + structured summary + recent tail
```

### 6.1 为什么保留 head

入口：[_protect_head_size](../agent/context_compressor.py#L1445)

head 通常包含：

- system prompt。
- 开头几条重要交互。
- 一些初始化约束或任务背景。

system prompt 永远隐式保护。`protect_first_n` 只统计 system 之后的非 system 消息。

### 6.2 为什么保留 tail

入口：[_find_tail_cut_by_tokens](../agent/context_compressor.py#L1505)

tail 是最近上下文，通常包含：

- 用户最新请求。
- 最近工具调用和结果。
- 当前工作状态。
- 刚发生的错误。

tail 不是按“固定 N 条消息”保护，而是按 token budget 从后往前累计。这样一个巨大工具结果不会因为“只算一条消息”而撑爆上下文。

额外规则：最新用户消息必须留在 tail。否则它会被写进 summary，而 `SUMMARY_PREFIX` 又告诉模型只响应 summary 后面的用户消息，当前任务就会消失。

### 6.3 为什么先裁剪旧工具输出

入口：[_prune_old_tool_results](../agent/context_compressor.py#L644)

工具输出可能非常大，比如：

- `rg` 搜索结果。
- `pytest` 长日志。
- 文件全文。
- 浏览器快照。
- computer-use 截图 base64。

直接喂给 summary model 会浪费上下文，也可能导致辅助模型请求失败。所以压缩前先做本地裁剪：

- 重复工具结果去重。
- 旧工具结果替换成一行摘要。
- 旧 tool_call 参数里的大 JSON 字符串合法截短。
- 历史多模态图片替换成文本占位。

这里的原则是：**旧工具输出不必完整保留，但必须留下“做过什么、对哪个文件/命令、结果如何”的线索。**

### 6.4 为什么要结构化 summary

入口：[_generate_summary](../agent/context_compressor.py#L1041)

summary 是给未来模型看的交接单，不是给人看的聊天总结。它固定包含：

- `Active Task`：当前最新未完成请求。
- `Goal`：整体目标。
- `Constraints & Preferences`：用户约束和偏好。
- `Completed Actions`：已完成动作，避免重复做。
- `Active State`：当前文件、分支、测试、进程等状态。
- `In Progress`：压缩发生时正在做什么。
- `Blocked`：错误和阻塞。
- `Key Decisions`：关键技术决策。
- `Resolved Questions`：已经回答的问题。
- `Pending User Asks`：尚未满足的请求。
- `Relevant Files`：相关文件。
- `Remaining Work`：剩余事项。
- `Critical Context`：不能丢的具体值、错误、配置。

这个结构重，是因为 agent 场景里“继续工作”比“知道大意”更重要。

### 6.5 为什么有 SUMMARY_PREFIX

入口：[SUMMARY_PREFIX](../agent/context_compressor.py#L26)

summary 会作为一条普通消息重新放进 messages。模型可能误读其中的 `Active Task` 或历史用户原话，把旧请求当作新请求。

所以 `SUMMARY_PREFIX` 明确说明：

- 这是背景，不是当前指令。
- 不要回答 summary 里的历史问题。
- 只响应 summary 后面的最新 user message。
- 如果最新 user message 和 summary 冲突，以最新 user message 为准。

这是压缩后不“重复旧任务”的关键防线。

### 6.6 为什么要修复 tool_call/tool_result

入口：[_sanitize_tool_pairs](../agent/context_compressor.py#L1391)

OpenAI 格式消息里，assistant 的 tool_call 和后续 tool result 必须配对。如果压缩切掉了其中一边，provider 可能直接 400。

压缩后会出现两类问题：

1. tool result 还在，但对应 assistant tool_call 被摘要掉了。
2. assistant tool_call 还在，但 tool result 被摘要掉了。

修复策略：

- 删除孤儿 tool result。
- 给缺失结果的 tool_call 补一个短占位 tool result。

这样压缩后的消息序列仍然是 provider 可接受的合法格式。

## 7. 显式 @ 上下文引用

入口：[context_references.py](../agent/context_references.py#L66)

用户可以在消息里写：

```text
@file:path.py
@file:path.py:10-30
@folder:src
@diff
@staged
@git:3
@url:https://example.com
```

处理链路：

```text
preprocess_context_references(...)
  -> preprocess_context_references_async(...)
    -> parse_context_references(message)
    -> _expand_reference(...)
       -> _expand_file_reference(...)
       -> _expand_folder_reference(...)
       -> _expand_git_reference(...)
       -> _fetch_url_content(...)
    -> 估算 injected_tokens
    -> 超过 context_length 50% 硬限制则拒绝
    -> 超过 context_length 25% 软限制则警告
    -> 从用户原消息移除 @ token
    -> 追加 Context Warnings / Attached Context
```

设计原则：

1. **显式引用优先**
   - 用户写了 `@file`，说明这段上下文本轮重要。
   - 它会被直接注入 user message，而不是等待模型自己搜索。

2. **注入必须限流**
   - 用户可能引用巨大文件或目录。
   - 超过上下文窗口 50% 会拒绝，超过 25% 会警告。

3. **路径必须受限**
   - 默认只允许当前工作目录内路径。
   - `.ssh`、`.aws`、`.gnupg`、Hermes `.env` 等敏感路径会被拒绝。

4. **二进制不直接注入**
   - 二进制文件无法安全变成文本上下文。

## 8. 插件上下文引擎

入口：[plugins/context_engine/__init__.py](../plugins/context_engine/__init__.py#L23)

Hermes 支持两类上下文引擎扩展：

1. 仓库内置：
   - 路径：`plugins/context_engine/<name>/`
   - 由 [discover_context_engines](../plugins/context_engine/__init__.py#L23) 和 [load_context_engine](../plugins/context_engine/__init__.py#L64) 扫描加载。

2. 用户通用插件：
   - 通过 `hermes_cli.plugins.get_plugin_context_engine()` 提供。

内置插件可以有两种写法：

```python
def register(ctx):
    ctx.register_context_engine(engine)
```

或者直接暴露一个 `ContextEngine` 子类，由加载器实例化。

`_EngineCollector` 是一个假的插件 ctx。它只收集 context engine，也允许上下文引擎注册自己的 slash command。命令会转发到全局插件命令注册表，并避免覆盖内置命令或普通插件命令。

## 9. 关键调用关系图

```text
AIAgent.__init__
  -> agent_init 选择 context engine
  -> context_engine.on_session_start

AIAgent.run_conversation
  -> conversation_loop
    -> build_turn_context
      -> 预检压缩 agent._compress_context
      -> 插件 pre_llm_call
      -> memory prefetch
    -> provider API call
    -> context_engine.update_from_response
    -> context_engine.should_compress
    -> agent._compress_context
      -> conversation_compression.compress_context
        -> context_engine.compress
        -> session DB 轮换
        -> memory/context_engine session switch
```

## 10. 关键状态字段

[ContextEngine](../agent/context_engine.py#L23) 要维护这些通用字段：

- `last_prompt_tokens`：最近一次真实输入 token。
- `last_completion_tokens`：最近一次输出 token。
- `last_total_tokens`：最近一次总 token。
- `threshold_tokens`：触发压缩的阈值。
- `context_length`：当前模型上下文窗口。
- `compression_count`：当前 session 压缩次数。

[ContextCompressor](../agent/context_compressor.py#L430) 还有一些实现细节字段：

- `_previous_summary`：上一次压缩 summary，用于迭代更新。
- `last_real_prompt_tokens`：最近 provider 返回的真实 prompt token。
- `last_compression_rough_tokens`：压缩后粗略估算。
- `awaiting_real_usage_after_compression`：刚压缩完，等待下一次真实 usage。
- `_ineffective_compression_count`：低收益压缩次数，防止 no-op 循环。
- `_summary_failure_cooldown_until`：摘要失败后的冷却时间。

## 11. 最容易误解的点

1. **`context_compressor` 不一定是 `ContextCompressor`**
   - 属性名是历史遗留。
   - 实际可能是任意 `ContextEngine` 插件。

2. **压缩不是把历史塞进 system prompt**
   - 压缩结果是一条 handoff summary 消息。
   - system prompt 会重建，但不承载完整历史。

3. **completion tokens 不该驱动压缩**
   - 下一轮输入窗口主要受 prompt tokens 影响。
   - reasoning 模型的 completion tokens 可能很大，但不等价于下一轮上下文压力。

4. **预检压缩和轮后压缩信号不同**
   - 预检没有真实 usage，只能粗略估算。
   - 模型返回后优先信任 provider 真实 prompt tokens。

5. **压缩会轮换 session_id**
   - 旧 session 以 `compression` 结束。
   - 新 session 继承标题和 parent_session_id。
   - memory provider 和 context engine 都会收到切换通知。

6. **最新用户消息必须留在 tail**
   - 如果最新请求被摘要掉，模型可能看不到“当前任务”。
   - 这是 agent 继续工作时最致命的一类上下文错误。

## 12. 相关测试入口

建议边看代码边看测试：

- [tests/agent/test_context_engine.py](../tests/agent/test_context_engine.py)
- [tests/agent/test_context_engine_host_contract.py](../tests/agent/test_context_engine_host_contract.py)
- [tests/agent/test_context_compressor.py](../tests/agent/test_context_compressor.py)
- [tests/agent/test_context_compressor_temporal_anchoring.py](../tests/agent/test_context_compressor_temporal_anchoring.py)
- [tests/agent/test_context_compressor_cross_session_guard.py](../tests/agent/test_context_compressor_cross_session_guard.py)
- [tests/agent/test_context_compressor_summary_continuity.py](../tests/agent/test_context_compressor_summary_continuity.py)
- [tests/agent/test_context_references.py](../tests/agent/test_context_references.py)
- [tests/agent/test_turn_context.py](../tests/agent/test_turn_context.py)
- [tests/run_agent/test_plugin_context_engine_init.py](../tests/run_agent/test_plugin_context_engine_init.py)
- [tests/run_agent/test_commit_memory_session_context_engine.py](../tests/run_agent/test_commit_memory_session_context_engine.py)
- [tests/gateway/test_compress_plugin_engine.py](../tests/gateway/test_compress_plugin_engine.py)
