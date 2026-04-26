# FastAgent

轻量级 ReAct Agent 框架 —— 基于 FSM 状态机 + 生命周期钩子，专为 LLM Agent 应用设计。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.5-orange)]()
[![Status](https://img.shields.io/badge/status-Alpha-yellow)]()

## 特性

- **FSM 状态机驱动** — 将 Agent 一轮对话拆解为 7 个标准状态（UserInput → LLMOutput → AfterLLMOutput → BeforeExecuteTools → ExecutingTools → AfterExecuteTools → AfterFinish），状态可插拔、可扩展
- **8 大生命周期钩子** — Before/After 级别的细粒度钩子，支持在 Agent 执行流程的任意阶段注入自定义逻辑
- **流式输出** — 支持 `chunk`（逐 token）和 `message`（完整消息）两种流式模式，完整输出思维链（reasoning_content）、文本内容和工具调用
- **工具系统** — 装饰器式 `@tool_creator` 定义工具，内置三类工具策略：普通工具、AskHuman（向用户提问）、Guard（敏感操作人工审批）
- **快照与中断恢复** — Agent 执行中可随时中断并序列化完整状态快照（Snapshot），后续反序列化恢复执行，适用于断联重连、人工审批等场景
- **MCP 集成** — 原生支持 Model Context Protocol，可通过 JSON 配置文件批量注册 MCP Server 工具，并支持对 MCP 工具注入 Guard 策略
- **沙箱执行** — 集成 OpenSandbox，Agent 可在隔离的沙箱容器中执行 Shell 命令
- **多模态支持** — 支持图片输入（本地 base64 / 公网 URL），工具可返回多模态内容（图文混合）
- **Embeddings** — 内置 Embedding 模块，支持 OpenAI 等兼容接口
- **JSON Output** — 支持结构化 JSON 输出（response_format）
- **多 Provider** — 已适配 DeepSeek、OpenAI、豆包 Seed，通过 AdapterFactory 可扩展

## 安装

```bash
pip install fast-agent
```

或从源码安装：

```bash
git clone https://github.com/Phantom-Algo/FastAgent.git
cd FastAgent
pip install -e .
```

依赖：`openai >= 2.21.0`, `pydantic >= 2.0.0`, `httpx >= 0.27.0`, `mcp >= 1.26.0`, `opensandbox-server >= 0.1.7`

## 快速开始

```python
import asyncio
from fast_agent import Agent, Context, LLMConfig, UserMessage, AssistantMessageOutputEvent

async def main():
    llm_config = LLMConfig(
        model_name="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com",
        provider="deepseek",
    )

    context = Context(
        system_prompt="你是一个乐于助人的AI助手。"
    )

    agent = Agent(llm_config=llm_config, context=context)

    async for event in agent.stream(
        UserMessage(content="你好，请介绍一下你自己"),
        stream_mode="chunk",
    ):
        if isinstance(event, AssistantMessageOutputEvent):
            print(f"Assistant: {event.data.content}")

asyncio.run(main())
```

## 核心概念

### Agent 状态机

Agent 的一轮对话被建模为有限状态机（FSM），依次经过以下状态：

```
AfterUserInput → LLMOutput → AfterLLMOutput → BeforeExecuteTools
                                                      ↓
                                               ExecutingTools
                                                      ↓
                                            AfterExecuteTools
                                                      ↓
                                                AfterFinish
```

每个状态都是一个独立的状态类，可以被子类化扩展。如果 LLM 无需调用工具，则直接从 AfterLLMOutput 跳转到 AfterFinish。

### 生命周期钩子

框架提供 8 个生命周期接口，每个接口对应一个执行阶段：

| 钩子 | 时机 | 说明 |
|------|------|------|
| `IAfterUserInput` | 用户输入之后 | 预处理用户消息 |
| `ILLMOutput` | LLM 输出阶段 | 调用 LLM 获取响应 |
| `IAfterLLMOutput` | LLM 输出之后 | 后处理 LLM 响应 |
| `IBeforeExecuteTools` | 工具执行之前 | 校验/过滤工具调用 |
| `IExecutingTools` | 工具执行阶段 | 执行工具调用（可注入 AskHuman 交互） |
| `IAfterExecuteTools` | 工具执行之后 | 处理工具执行结果 |
| `IAfterFinish` | 轮次结束之后 | 清理、日志等收尾工作 |

通过 `LifespanManager` 注册自定义钩子实现：

```python
from fast_agent import LifespanManager, DefaultExecutingTools

class MyToolExecutor(DefaultExecutingTools):
    async def execute(self, data):
        # 自定义工具执行逻辑，比如注入 AskHuman 交互
        return await super().execute(data)

lifespan = LifespanManager(executing_tools=MyToolExecutor())
agent = Agent(llm_config=llm_config, context=context, lifespan_manager=lifespan)
```

### 事件系统

Agent 通过异步生成器产出事件，客户端按类型处理：

```python
from fast_agent import (
    AssistantMessageChunkOutputEvent,  # 流式 chunk（reasoning/content/refusal）
    AssistantMessageOutputEvent,       # 完整 Assistant 消息
    ToolCallEvent,                     # 工具调用检测
    ToolsExecutedEvent,                # 工具执行完成
    GuardTriggeredEvent,               # Guard 触发（需人工审批）
    InterruptEvent,                    # 中断事件（含 Snapshot）
    RoundStopEvent,                    # 当前轮次结束
    AskHumanEvent,                     # 工具向用户提问
    AskHumanResponseEvent,             # 用户对提问的响应
)
```

## 工具系统

使用 `@tool_creator` 装饰器定义工具，框架自动从函数签名生成 JSON Schema 参数模型。

### 普通工具

```python
from fast_agent import tool_creator

@tool_creator(
    tool_name="get_weather",
    tool_description="查询指定城市的天气信息",
    labels=["weather"],
)
async def get_weather(city: str) -> str:
    return f"{city}: 晴 22°C"

context = Context(
    system_prompt="你是一个天气助手",
    tool_manager=[get_weather],
)
```

### AskHuman 工具

工具执行过程中向用户发起提问，获取用户偏好后继续执行：

```python
from fast_agent import tool_creator, AskHumanPolicy
from fast_agent.types.tool.base_tool_runtime import BaseToolRuntime

@tool_creator(
    tool_name="book_hotel",
    tool_description="预订酒店",
    ask_human_policy=AskHumanPolicy(timeout=120),
)
async def book_hotel(city: str, tool_runtime: BaseToolRuntime = None) -> str:
    response = await tool_runtime.ask_human(
        data={"question": f"请选择 {city} 的房型: 1)标准间 2)大床房 3)套房"},
        timeout=120,
    )
    choice = response.get("answer", "1")
    return f"预订成功: 房型 {choice}"
```

### Guard 工具

敏感操作触发人工审批，审批期间 Agent 状态通过 Snapshot 持久化，审批后恢复执行：

```python
from fast_agent import tool_creator, GuardPolicy, GuardPolicyHumanResponseSchema, ToolResultMessage

class PaymentApproval(GuardPolicyHumanResponseSchema):
    approved: bool = False
    reason: str = ""

@tool_creator(
    tool_name="process_payment",
    tool_description="处理支付",
    guard_policy=GuardPolicy(
        info="支付操作需要审批",
        schema=PaymentApproval(),
        guard_func=lambda r: r.approved,
        reject_func=lambda r: ToolResultMessage(
            tool_call_id="_", name="process_payment",
            content=f"已拒绝: {r.reason}", is_error=False,
        ),
    ),
)
async def process_payment(item: str, amount: str) -> str:
    return f"支付成功: {item} {amount}"
```

Guard 触发后，Agent 产出 `GuardTriggeredEvent`（含 Snapshot），客户端收集用户审批后调用 `agent.resume_stream(snapshot, human_response)` 恢复执行。

## 快照与中断恢复

Agent 在执行中可随时被中断（如客户端断联），并自动生成状态快照：

```python
# 中断
agent.request_interrupt(reason="client_disconnect")

# 处理中断事件
async for event in agent.stream(user_msg):
    if isinstance(event, InterruptEvent):
        snapshot = event.data.snapshot
        # 序列化保存
        data = snapshot.serialize()
        # 后续反序列化恢复
        restored = Snapshot.deserialize(data)

# 恢复执行
async for event in agent.resume_stream(restored, stream_mode="chunk"):
    # 处理恢复后的事件流
```

## MCP 集成

通过 JSON 配置文件批量注册 MCP Server 工具，支持对 MCP 工具注入 Guard 策略：

```python
from fast_agent import MCPManager, GuardPolicy

manager = MCPManager()

# 从 JSON 配置文件注册 MCP 工具
tools = await manager.register_servers_from_addresses(["mcpServer.json"])

# 对指定工具注入 Guard 策略
manager.enhance_tool_by_id(
    id=tools[0].id,
    guard_policy=GuardPolicy(
        info="MCP 工具需审批",
        schema=MyApproval(),
        guard_func=lambda r: r.approved,
        reject_func=my_reject,
    ),
)

context = Context(
    system_prompt="你是 MCP 工具助手",
    tool_manager=manager.get_tools(),
)
```

MCP 配置文件示例 (`static/json/mcpServer.json`)：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-brave-search"],
      "env": { "BRAVE_API_KEY": "your-key" }
    }
  }
}
```

## 沙箱执行

集成 OpenSandbox，让 Agent 在隔离沙箱中执行 Shell 命令：

```python
from fast_agent import OpenSandboxFactory, CommandOpts
from datetime import timedelta

factory = OpenSandboxFactory()
sandbox = await factory.create_sandbox(
    image="ubuntu:22.04",
    domain="localhost:8080",
    api_key="your-api-key",
    timeout=timedelta(minutes=10),
)

result = await sandbox.command(
    "ls -la /",
    opts=CommandOpts(timeout=timedelta(seconds=30)),
)
print(result.logs.stdout)
```

## 多模态（图片）支持

支持在 UserMessage 和 ToolResultMessage 中传递多模态内容：

```python
from fast_agent import UserMessage, TextPart, ImagePart

user_msg = UserMessage(content=[
    TextPart(text="请分析这张架构图"),
    ImagePart(
        base64_data="iVBORw0KGgo...",  # base64 编码的图片
        mime_type="image/png",
        detail="high",
    ),
])
```

工具也可以返回混合图文内容：

```python
@tool_creator(tool_name="get_diagram", tool_description="获取参考图")
async def get_diagram() -> list:
    return [
        TextPart(text="参考架构图如下："),
        ImagePart(base64_data="...", mime_type="image/png"),
    ]
```

## Embeddings

```python
from fast_agent import AdapterFactory, EmbeddingConfig

config = EmbeddingConfig(
    model_name="text-embedding-3-small",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    provider="openai",
)

adapter = AdapterFactory.get_adapter_cls(config.provider)()
response = await adapter.embed(config, ["hello", "world"])
print(f"维度: {len(response.data[0].embedding)}")
```

## JSON Output

通过 `LLMConfig` 的 `response_format` 参数启用结构化 JSON 输出：

```python
llm_config = LLMConfig(
    model_name="deepseek-chat",
    api_key="your-api-key",
    base_url="https://api.deepseek.com",
    provider="deepseek",
    response_format={"type": "json_object"},
)
```

## MessageManager - 对话管理

提供完整的消息和 Round 增删查改 API：

```python
from fast_agent import MessageManager, UserMessage, AssistantMessage

manager = MessageManager()

# 添加消息
manager.add_message(UserMessage(content="你好"))
manager.add_message(AssistantMessage(content="你好！有什么可以帮你的？"))

# 查询 Round
rounds = manager.get_rounds(start_round_index=0, end_round_index=2)
print(f"共 {manager.get_round_count()} 轮对话")

# 更新消息
manager.update_message_by_id(msg_id, new_message)

# 删除指定 Round
removed = manager.remove_rounds(0, 1)
```

## 支持的 Provider

| Provider | 标识符 | 模型示例 |
|----------|--------|---------|
| DeepSeek | `deepseek` | deepseek-v4-flash, deepseek-v4-pro |
| OpenAI | `openai` | gpt-4o, gpt-4-turbo |
| 豆包 Seed | `doubao_seed` | doubao-seed-2-0-pro |

## 项目结构

```
FastAgent/
├── src/fast_agent/
│   ├── types/                  # 抽象基类、接口和数据模型
│   │   ├── adapter/            #   Adapter 抽象
│   │   ├── agent/              #   Agent FSM / Event / Lifespan / Snapshot 抽象
│   │   ├── context/            #   Context 抽象
│   │   ├── embeddings/         #   Embedding 抽象与领域模型
│   │   ├── llm/                #   LLM Config 抽象与 Provider 枚举
│   │   ├── mcp/                #   MCP 抽象
│   │   ├── messages/           #   Message / Chunk / Round 抽象与领域模型
│   │   ├── sandbox/            #   Sandbox 抽象
│   │   ├── system_prompt/      #   SystemPrompt 抽象
│   │   └── tool/               #   Tool / Guard / AskHuman 抽象与领域模型
│   └── resources/              # 具体实现
│       ├── adapter/            #   DeepSeek / OpenAI / 豆包 Adapter
│       ├── agent/              #   Agent / FSM / Event / Lifespan / Snapshot 实现
│       ├── context/            #   Context 实现
│       ├── embeddings/         #   EmbeddingConfig 实现
│       ├── llm/                #   LLMConfig 实现
│       ├── mcp/                #   MCP Manager / Adapter 实现
│       ├── messages/           #   MessageManager 实现
│       ├── sandbox/            #   OpenSandbox Factory / Instance 实现
│       ├── system_prompt/      #   SystemPrompt 实现
│       └── tool/               #   Tool / ToolCreator / ToolManager 实现
├── examples/                   # 示例
│   ├── demo_deepseek.py        #   综合演示：流式输出 + 工具 + Guard + Snapshot
│   ├── demo_mcp.py             #   MCP 工具注册 + Guard 增强
│   ├── demo_sandbox.py         #   沙箱命令执行
│   └── demo_image_recognition.py  # 多模态图片识别
├── tests/                      # 测试
└── static/                     # 静态资源（架构图、配置文件、文档）
```

## License

MIT © Phantom-Algo
