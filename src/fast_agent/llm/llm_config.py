from pydantic import BaseModel, ConfigDict
from typing import List

class LLMConfig(BaseModel):
    """
    LLMConfig 用于配置模型选择与生成参数，包含以下字段：

    - 模型配置
        - model: 模型名称
        - api_key: API 密钥
        - base_url: API 基础 URL

    - 模型生成配置
        - temperature: 采样温度
            - 值越高，生成的文本越随机（适合创造性任务）
            - 值越低，生成的文本越确定性（适合精确、事实性任务）
        - top_p: 采样的 top_p 值
            - 控制生成文本的多样性
            - 值越高，生成的文本越多样化
            - 值越低，生成的文本越集中在高概率的选项上
        - max_tokens: 最大生成的 token 数量
        - frequency_penalty: 频率惩罚系数
            - 用于减少重复内容的生成
            - 值越高，模型越不倾向于重复使用相同的词汇
        - presence_penalty: 存在惩罚系数
            - 用于鼓励模型引入新的话题或概念
            - 值越高，模型越倾向于引入新的内容
        - stop_sequences: 停止生成的序列列表
            - 当生成的文本包含这些序列时，模型将停止生成
        - tool_choice: 工具选择模式
            - "auto": 模型自动决定是否使用工具
            - "none": 不使用任何工具
            - "required": 强制使用工具（至少一个或多个）
            - 其他字符串: 指定特定的工具名称
        - parallel_tool_calls: 是否允许并行调用工具
    """

    # 模型配置
    model_name: str
    api_key: str
    base_url: str

    # 模型生成配置
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = []
    tool_choice: str = "auto"
    parallel_tool_calls: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)