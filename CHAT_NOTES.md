# Hermes Study - 代码分析笔记

## 对话记录

### 问题1：`agent_init.py` 第1270行代码分析

**原始代码**：
```python
compression_threshold = float(_compression_cfg.get("threshold", 0.50))
```

**位置**：`agent/agent_init.py` 第1270行

**三行代码工具注册**（第1576-1580行）：
```python
if _tname:
    agent.valid_tool_names.add(_tname)
    agent._context_engine_tool_names.add(_tname)
    _existing_tool_names.add(_tname)
```

---

## 知识总结

### 1. 工具名称追踪集合的作用

#### 三个集合的职责

| 集合名称 | 用途 | 使用场景 |
|---------|------|--------|
| `valid_tool_names` | 所有工具的全局注册表 | 系统提示生成、模型调用时的工具验证 |
| `_context_engine_tool_names` | 上下文引擎工具专用追踪 | 引擎卸载时清理、压缩状态恢复、工具过滤 |
| `_existing_tool_names` | 本轮初始化的临时去重 | 防止相同工具名注册多次 |

#### 为什么要区分内置工具 vs 上下文引擎工具

**根本原因**：生命周期和职责不同

| 维度 | 内置工具 | 上下文引擎工具 |
|-----|---------|-----------------|
| **生命周期** | 整个会话生命周期 | 引擎的生命周期（可切换/禁用） |
| **职责** | 执行具体工作任务 | 管理上下文窗口 |
| **例子** | terminal, read_file, write_file | lcm_grep, lcm_describe, lcm_expand |
| **系统提示** | 需要专门指导 | 透明基础设施 |
| **压缩处理** | 无特殊处理 | 状态重置、迭代更新 |

**实际应用**：
- 内置工具的指导通过 `system_prompt.py` 中的条件检查注入
- 上下文引擎工具由引擎自行决定是否暴露
- 压缩/切换引擎时，需要清理旧引擎的工具，注册新引擎的工具

---

### 2. 上下文引擎（Context Engine）系统

#### 什么是上下文引擎

**定义**：负责在会话接近模型上下文窗口上限时，决定**如何压缩对话、暴露什么工具、如何管理上下文**的可插拔系统。

从 `context_engine.py` 的抽象基类定义：
- 决定是否压缩和何时压缩
- 实施具体的压缩策略
- 暴露引擎专属工具
- 管理会话生命周期事件

#### 引擎选择流程

**优先级顺序**（`agent_init.py` 第1465-1502行）：

```
优先级 1: 读 config.yaml 中的 context.engine 配置
           ↓
优先级 2: 从仓库内置插件加载 (plugins/context_engine/<name>/)
           ↓
优先级 3: 从用户通用插件加载 (~/.hermes/plugins/)
           ↓
优先级 4: 回退到内置 ContextCompressor
```

#### 三种引擎来源

| 来源 | 位置 | 说明 |
|------|------|------|
| **内置引擎** | `agent/context_compressor.py` | `ContextCompressor` - 有损摘要策略 |
| **仓库插件** | `plugins/context_engine/<name>/` | 项目内置的自定义引擎 |
| **用户插件** | `~/.hermes/plugins/` | 用户通过插件系统安装的引擎 |

#### ContextEngine 必须实现的接口

```python
class ContextEngine(ABC):
    # 核心接口（必须实现）
    @property
    @abstractmethod
    def name(self) -> str:
        """返回引擎短名称"""
    
    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """根据模型响应更新 token 状态"""
    
    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """判断是否达到压缩阈值"""
    
    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """压缩消息列表"""
    
    # 可选接口
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回引擎专属工具"""
        return []
    
    def handle_tool_call(self, name: str, args: Dict) -> str:
        """处理工具调用"""
    
    def on_session_start(self, session_id: str, **kwargs) -> None:
        """会话开始时调用"""
```

#### 配置示例

**使用内置引擎（默认）**：
```yaml
# config.yaml
context:
  engine: "compressor"      # 默认值，可省略
  threshold: 0.50           # 50% 时触发压缩
  target_ratio: 0.20        # 压缩后保留 20%
  enabled: true
```

**使用自定义引擎**：
```yaml
context:
  engine: "my_custom_engine"  # 自定义引擎名称
```

---

### 3. 代码执行流程

#### 工具加载全流程

```
第1阶段: 加载基础工具集合
├─ agent.tools = get_tool_definitions(...)
├─ agent.valid_tool_names = {所有工具名称}
└─ 状态：_existing_tool_names = 所有工具名

第2阶段: 注入内存提供者工具
├─ for _schema in agent._memory_manager.get_all_tool_schemas():
├─ if _tname not in _existing_tool_names:
│  ├─ agent.tools.append(_wrapped)
│  └─ agent.valid_tool_names.add(_tname)
└─ 状态：_context_engine_tool_names = {} (未设置)

第3阶段: 注入上下文引擎工具 ⭐ 关键
├─ agent._context_engine_tool_names = set()  # 初始化追踪集
├─ for _schema in agent.context_compressor.get_tool_schemas():
├─ if _tname not in _existing_tool_names:
│  ├─ agent.tools.append(_wrapped)
│  ├─ agent.valid_tool_names.add(_tname)
│  ├─ agent._context_engine_tool_names.add(_tname)  # ⭐ 标记
│  └─ _existing_tool_names.add(_tname)
└─ 结果：清晰区分了引擎工具
```

#### 压缩触发时的处理

```
压缩开始
├─ agent.context_compressor.compress(messages, ...)
├─ 调用 _generate_summary() 生成摘要
├─ 迭代更新 _previous_summary
└─ 返回压缩后的消息列表

压缩后处理
├─ on_session_start() 不调用（同一会话）
├─ _context_engine_tool_names 保持不变
└─ 内存工具/内置工具都保持可用
```

---

### 4. 关键代码位置对应表

| 功能 | 文件位置 | 行号 |
|------|---------|------|
| 内置工具加载 | `agent_init.py` | 948-967 |
| 内存工具注入 | `agent_init.py` | 1209-1225 |
| 引擎选择逻辑 | `agent_init.py` | 1465-1502 |
| 上下文引擎工具注入 | `agent_init.py` | 1554-1580 |
| 引擎会话开始 | `agent_init.py` | 1582-1594 |
| 系统提示中的工具指导 | `system_prompt.py` | 111-183 |
| ContextEngine 接口定义 | `context_engine.py` | 23-138 |
| ContextCompressor 实现 | `context_compressor.py` | 422+ |
| 压缩工具结果处理 | `context_compressor.py` | 621-751 |
| 摘要生成 | `context_compressor.py` | 1012-1287 |

---

### 5. 压缩阈值相关

**第1270行的含义**：
```python
compression_threshold = float(_compression_cfg.get("threshold", 0.50))
```

- 从配置读取压缩阈值（默认 0.50 = 50%）
- 当对话占用上下文窗口 50% 时，触发压缩
- 这是**内置 ContextCompressor** 的配置
- **自定义引擎**可能有不同的压缩逻辑，此参数对其不适用

**后续处理**（第1286-1305行）：
```python
_model_cthresh = _cthresh_fn(agent.model, agent.provider, ...)
if _model_cthresh is not None:
    compression_threshold = _model_cthresh  # 模型特定的覆盖
```

- 某些模型（如 Codex gpt-5.5）有特殊的压缩阈值
- GPT-5.5 的阈值自动提升到 85%
- 用户可通过 `compression.codex_gpt55_autoraise: false` 禁用

---

## 去重机制详解

### 为什么需要去重

在工具注入的三个阶段，同一个工具名可能出现多次：
1. 内置工具中可能已有 `memory` 工具
2. 内存提供者可能注册相同的 `memory` 工具
3. 上下文引擎可能注册工具

**去重流程**：
```python
_existing_tool_names = {已有工具名}

if _tname and _tname in _existing_tool_names:
    continue  # 跳过，避免重复

agent.tools.append(_wrapped)
agent.valid_tool_names.add(_tname)
_existing_tool_names.add(_tname)  # 记录已添加
```

### 不同来源工具的优先级

| 优先级 | 来源 | 处理方式 |
|--------|------|--------|
| 1 | 内置工具集 | 直接添加，无重复 |
| 2 | 插件/内存工具 | 检查后添加（如果不存在） |
| 3 | 上下文引擎工具 | 检查后添加（如果不存在） |

---

## 实践建议

### 何时会用到 `_context_engine_tool_names`

1. **引擎切换时**：清理旧引擎的工具
   ```python
   # 从 compressor 切换到 vcm_engine
   if old_engine_tools:
       agent.tools = [t for t in agent.tools 
                      if t["function"]["name"] not in old_engine._context_engine_tool_names]
   ```

2. **压缩后状态恢复**：保留引擎工具可用性
   ```python
   # 确保压缩后引擎工具仍可用
   if tool_name in agent._context_engine_tool_names:
       # 工具来自引擎，处理特殊逻辑
   ```

3. **日志和调试**：追踪哪些工具来自哪里
   ```python
   logger.info(f"Context engine tools: {agent._context_engine_tool_names}")
   logger.info(f"Total tools: {len(agent.valid_tool_names)}")
   ```

---

## 相关文件导航

- **核心初始化**：`agent/agent_init.py`
- **引擎接口**：`agent/context_engine.py`
- **默认引擎**：`agent/context_compressor.py`
- **系统提示**：`agent/system_prompt.py`
- **会话循环**：`agent/conversation_loop.py`

---

**最后更新**：2026-06-18
