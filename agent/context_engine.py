"""可插拔上下文引擎接口。

上下文引擎负责在会话接近模型上下文窗口上限时，决定是否压缩、怎样压缩、
是否暴露专属工具，以及如何记录模型真实返回的 token 使用量。

默认实现是 ``ContextCompressor``；第三方实现可以通过通用插件系统或
``plugins/context_engine/<name>/`` 目录接入。配置项 ``context.engine``
决定当前启用哪个引擎，同一时间只会有一个引擎生效。

生命周期：
1. 初始化并注册引擎。
2. 会话开始时调用 ``on_session_start``。
3. 每次模型响应后调用 ``update_from_response`` 更新 token 统计。
4. 每轮后调用 ``should_compress`` 判断是否需要压缩。
5. 需要压缩时调用 ``compress`` 返回新的 OpenAI 消息列表。
6. 真正的会话边界调用 ``on_session_end``，不会每轮调用。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """所有上下文引擎都必须实现的抽象基类。"""

    # -- Identity ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """返回引擎短名称，例如 ``compressor`` 或 ``lcm``。"""

    # token 状态会被 run_agent.py / conversation_loop.py 直接读取，用于展示和日志。

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # 预检压缩参数：system prompt 总是隐式保护，protect_first_n 只统计非 system 的头部消息。

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- Core interface ----------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """根据模型响应里的标准化 usage 字典更新 token 状态。"""

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """判断当前轮是否达到压缩阈值。"""

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """压缩消息列表并返回新的 OpenAI 格式消息序列。"""

    # -- Optional: pre-flight check ----------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """API 调用前的粗略压缩预检；默认不启用。"""
        return False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """粗略估算明显偏噪时，是否改信最近一次真实 provider usage。"""
        return False

    # -- Optional: manual /compress preflight ------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """供手动 ``/compress`` 使用：当前消息里是否存在可压缩中间区。"""
        return True

    # -- Optional: session lifecycle ---------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """会话开始或压缩轮换 session_id 后调用，可加载/继承引擎状态。"""

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """真实会话结束时调用，用于落盘、释放连接或清理内存状态。"""

    def on_session_reset(self) -> None:
        """``/new`` 或 ``/reset`` 时重置每个会话独有的计数和 token 状态。"""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    # -- Optional: tools ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回该上下文引擎额外暴露给模型调用的工具 schema。"""
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理上下文引擎专属工具调用，返回 JSON 字符串。"""
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Optional: status / display ----------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """返回展示和日志需要的上下文引擎状态。"""
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- Optional: model switch support ------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """模型切换或 fallback 激活时刷新上下文窗口和压缩阈值。"""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)
