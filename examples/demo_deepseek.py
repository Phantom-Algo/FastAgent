import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from fast_agent.llm import (
    LLMConfig, 
    Context, 
    AssistantMessage, 
    UserMessage, 
)
from fast_agent.agent import (
    Agent, 
    Lifespan, 
    IAfterFinish,
    InputAfterFinish,
    OutputAfterFinish,
    AssistantMessageChunkOutputEvent,
    ToolCallEvent,
    AssistantMessageOutputEvent,
    HumanReviewEvent,
    HumanResponseEvent,
    InterruptEvent
)
from fast_agent.tool import tool, ToolRuntime

# --- 模拟数据 ---
USER_DB = [
    {
        "id": "user_0001",
        "name": "Alice",
        "password": "password123"
    },
    {
        "id": "user_0002",
        "name": "Bob",
        "password": "securepass456"
    },
    {
        "id": "user_0003",
        "name": "Charlie",
        "password": "charlie789"
    }
]

CONTEXT = {
    "id": None,
    "name": None,
}

# --- DeepSeek 特殊处理 ---
class DeepSeekAfterFinish(IAfterFinish):
    async def after_finish(self, data: InputAfterFinish) -> OutputAfterFinish:
        context = data.context
        for work_message in context.work_messages.get_messages():
            if isinstance(work_message, AssistantMessage):
                work_message.reasoning_content = None

        return OutputAfterFinish(
            llm_config=data.llm_config,
            context=data.context,
            event_channel=data.event_channel,
            kwargs=data.kwargs
        )


def build_agent(context: Context, lifespan: Optional[Lifespan] = None) -> Agent:
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY，请在 .env 中配置后再运行。")

    llm_config = LLMConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        provider="deepseek"
    )

    if lifespan is None:
        lifespan = Lifespan()
    
    return Agent(
        llm_config=llm_config,
        context=context,
        lifespan=lifespan
    )

@tool(
    description="当查询到用户为未注册用户时，请求用户进行注册",
    human_review_timeout=20
)
async def request_registration(tool_runtime: ToolRuntime):
    # 发起人工审核请求
    try:
        res = await tool_runtime.human_review({
            "message": "系统检测到您尚未注册，请按照流程进行注册，谢谢配合。",
            "request": {
                "name": "请输入用户名",
                "password": "请输入密码"
            }
        }, timeout=tool_runtime.human_review_timeout)
    except TimeoutError as e:
        return f"用户未在{tool_runtime.human_review_timeout}秒内完成注册，注册请求已超时。"
    except Exception as e:
        return f"用户注册流程发生错误：{str(e)}"
    
    def _validate_registration_data(data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "name" not in data or "password" not in data:
            return False
        if not isinstance(data["name"], str) or not isinstance(data["password"], str):
            return False
        return True
    
    if not _validate_registration_data(res):
        return "用户注册数据格式错误，注册失败。"
    
    username_exists = any(user["name"] == res["name"] for user in USER_DB)
    # 模拟注册逻辑（实际应用中应替换为数据库操作等）
    if username_exists:
        # --- 第二次注册请求 ---
        try:
            res = await tool_runtime.human_review({
                "message": f"用户名 {res['name']} 已存在，请重新输入用户名。",
                "request": {
                    "name": "请输入用户名",
                    "password": "请输入密码"
                }
            }, timeout=tool_runtime.human_review_timeout)
        except TimeoutError as e:
            return f"用户未在{tool_runtime.human_review_timeout}秒内完成注册，注册请求已超时。"
        except Exception as e:
            return f"用户注册流程发生错误：{str(e)}"
        
        if not _validate_registration_data(res):
            return "用户注册数据格式错误，注册失败。"
        
        username_exists = any(user["name"] == res["name"] for user in USER_DB)
        if username_exists:
            return f"用户名 {res['name']} 已存在，注册失败。"
    
    new_user = {
        "id": f"user_{len(USER_DB) + 1:04d}",
        "name": res["name"],
        "password": res["password"]
    }
    USER_DB.append(new_user)
    return f"用户 {res['name']} 注册成功！"


@tool(
    description="查询用户当前注册状态"
)
def check_registration_status() -> str:
    if CONTEXT["id"] is None:
        return "当前用户为未注册用户，需注册后继续使用"
    else:
        return f"当前用户为已注册用户: {CONTEXT['name']}"

async def main():
    context = Context(
        system_prompt="你是一个智能问答助手，但是在用户第一次向你提问时需检查用户是否注册，若未注册则引导用户完成注册流程。若用户一直未注册，则不能够继续后续问答。",
        tools=[check_registration_status, request_registration]
    )

    agent = build_agent(context=context, lifespan=Lifespan(after_finish=DeepSeekAfterFinish()))

    print("\n======== 登录 ========\n")
    name = input("请输入用户名：")
    password = input("请输入密码：")
    if name and password:
        matched_user = next((user for user in USER_DB if user["name"] == name and user["password"] == password), None)
        if matched_user:
            CONTEXT["id"] = matched_user["id"]
            CONTEXT["name"] = matched_user["name"]
            print(f"登录成功！欢迎 {CONTEXT['name']}！")
        else:
            print("用户名或密码错误，继续以未注册用户身份进行对话。")
    else:        
        print("未输入用户名或密码，继续以未注册用户身份进行对话。")

    INTERRUPT: InterruptEvent = None
    pending_human_review_task: Optional[asyncio.Task] = None
    state: Literal["user_input", "reasoning", "tool_call", "content_output", "human_review", "round_stop"] = None

    async def _collect_human_review_and_respond(response_channel, request):
        name = await asyncio.to_thread(input, f"{request['name']}: ")
        password = await asyncio.to_thread(input, f"{request['password']}: ")

        if response_channel.done():
            print("‼️ 人工审核请求已超时或结束，输入结果已丢弃。")
            return

        response_channel.set_result(HumanResponseEvent(
            type="human_normal_response",
            data=HumanResponseEvent.HumanResponseEventData(
                is_success=True,
                message="用户已完成注册表单填写",
                content={
                    "name": name,
                    "password": password
                }
            )
        ))

    while True:
        streamer = None
        if INTERRUPT:
            streamer = agent.resume_stream(snapshot=INTERRUPT.data.snapshot)
        else:
            state = "user_input"
            print("\n======== 等待用户输入 ========\n")
            user_input = input("用户：")
            if user_input.lower() in ["exit", "quit"]:
                print("退出对话。")
                break
            streamer = agent.stream(user_input=UserMessage(content=user_input))
        
        async for event in streamer:
            if isinstance(event, AssistantMessageChunkOutputEvent):
                data = event.data
                if data.type == "reasoning_content":
                    if state != "reasoning":
                        print("\n======== Agent 思考中 ========\n")
                        state = "reasoning"
                    print(data.reasoning_content, end="", flush=True)
                
                elif data.type == "content":
                    if state != "content_output":
                        print("\n======== Agent 输出中 ========\n")
                        state = "content_output"
                    print(data.content, end="", flush=True)

                else:
                    if state != "content_output":
                        print("\n======== Agent 输出中 ========\n")
                        state = "content_output"
                    print(data.refusal, end="", flush=True)

            elif isinstance(event, ToolCallEvent):
                data = event.data
                if state != "tool_call":
                    print("\n======== Agent 调用工具中 ========\n")
                    state = "tool_call"
                print(f"Executing tool: 🔧 {data.function_name}")

            elif isinstance(event, HumanReviewEvent):
                data = event.data
                if state != "human_review":
                    print("\n======== Agent 发起人工审核请求 ========\n")
                    state = "human_review"
                content = data.content
                response_channel = data.response_channel
                message = content.get("message")
                request = content.get("request")
                print(f"Message: {message}")
                try:
                    if pending_human_review_task and not pending_human_review_task.done():
                        pending_human_review_task.cancel()
                    pending_human_review_task = asyncio.create_task(
                        _collect_human_review_and_respond(response_channel, request)
                    )
                except Exception as e:
                    print(f"‼️ 失败！无法发送人工审核响应: {str(e)}")

            elif isinstance(event, InterruptEvent):
                INTERRUPT = event
                print(f"\n======== Agent 执行被中断: {INTERRUPT.data.reason} ========\n")
                break

if __name__ == "__main__":
    asyncio.run(main())