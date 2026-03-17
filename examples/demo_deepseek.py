"""
FastAgent 综合演示 —— 智能旅行规划助手

场景说明：
    用户与一个智能旅行规划助手进行多轮对话。
    助手可以帮助用户查询天气、搜索航班、预订酒店、推荐餐厅、处理支付和修改行程。

工具分类（三分之一各一类）：
    ● 普通工具（2个）：get_weather, search_flights
    ● Ask Human 工具（2个）：book_hotel, recommend_restaurant
    ● Guard 工具（2个）：process_payment, modify_itinerary

演示功能：
    1. 循环对话（输入 quit/exit 退出）
    2. 流式输出（思维链 + 内容 + 工具调用）
    3. Ask Human 机制（工具向用户提问）
    4. Guard 机制（敏感操作需人工审批）
    5. Snapshot 序列化与反序列化
    6. 美观的控制台输出
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fast_agent.resources.agent.agent import Agent
from fast_agent.resources.agent.lifespan.lifespan_manager import LifespanManager
from fast_agent.resources.agent.lifespan.default_lifespan import DefaultExecutingTools
from fast_agent.resources.agent.event.events import (
    AskHumanEvent,
    AskHumanResponseEvent,
    AssistantMessageChunkOutputEvent,
    AssistantMessageOutputEvent,
    GuardTriggeredEvent,
    InterruptEvent,
    RoundStopEvent,
    ToolCallEvent,
    ToolsExecutedEvent,
)
from fast_agent.resources.agent.snapshot.snapshot import Snapshot
from fast_agent.resources.context.context import Context
from fast_agent.resources.llm.llm_config import LLMConfig
from fast_agent.resources.tool.tool_creator import tool_creator
from fast_agent.types.agent.lifespan.base_lifespan import IExecutingTools
from fast_agent.types.agent.lifespan.dto.lifespan_dto import (
    ExecutingToolsRequest,
    ExecutingToolsResponse,
)
from fast_agent.types.messages.domain.user_message import UserMessage
from fast_agent.types.messages.domain.tool_result_message import ToolResultMessage
from fast_agent.types.tool.base_tool_runtime import BaseToolRuntime
from fast_agent.types.tool.domain.ask_human_policy import AskHumanPolicy
from fast_agent.types.tool.domain.guard_policy import GuardPolicy, GuardPolicyHumanResponseSchema


# ================================================================
# ANSI 控制台样式工具
# ================================================================

class Style:
    RESET      = "\033[0m"
    BOLD       = "\033[1m"
    DIM        = "\033[2m"
    ITALIC     = "\033[3m"
    # 前景色
    RED        = "\033[31m"
    GREEN      = "\033[32m"
    YELLOW     = "\033[33m"
    BLUE       = "\033[34m"
    MAGENTA    = "\033[35m"
    CYAN       = "\033[36m"
    WHITE      = "\033[37m"
    GRAY       = "\033[90m"
    # 背景色
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"


def print_banner():
    banner = f"""
{Style.CYAN}{Style.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        ✈️  FastAgent 智能旅行规划助手  ✈️                    ║
║                                                              ║
║   基于 DeepSeek 大模型 · 流式输出 · 工具调用 · Guard机制     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET}
"""
    print(banner)


def print_separator(char="─", length=62, color=Style.GRAY):
    print(f"{color}{char * length}{Style.RESET}")


def print_info(msg: str):
    print(f"{Style.CYAN}ℹ {msg}{Style.RESET}")


def print_success(msg: str):
    print(f"{Style.GREEN}✔ {msg}{Style.RESET}")


def print_warning(msg: str):
    print(f"{Style.YELLOW}⚠ {msg}{Style.RESET}")


def print_error(msg: str):
    print(f"{Style.RED}✘ {msg}{Style.RESET}")


def print_tool_header(name: str, args: Dict[str, Any]):
    args_str = json.dumps(args, ensure_ascii=False, indent=None)
    print(f"\n{Style.YELLOW}{Style.BOLD}🔧 调用工具: {name}{Style.RESET}")
    print(f"{Style.GRAY}   参数: {args_str}{Style.RESET}")


def print_tool_result(results: list):
    for r in results:
        status = f"{Style.RED}错误" if r.is_error else f"{Style.GREEN}成功"
        content = r.content if isinstance(r.content, str) else json.dumps(r.content, ensure_ascii=False)
        print(f"{Style.YELLOW}   📋 [{r.name}] {status}{Style.RESET}: {Style.DIM}{content}{Style.RESET}")


def print_guard_prompt(contexts):
    print(f"\n{Style.RED}{Style.BOLD}🛡️  安全审批 (Guard 机制触发){Style.RESET}")
    print_separator("━", 62, Style.RED)
    for ctx in contexts:
        tc = ctx.tool_call
        tool = ctx.tool_info
        info = tool.guard_policy.info if tool and tool.guard_policy else "需要人工审批"
        print(f"  {Style.YELLOW}工具: {tc.function_name}{Style.RESET}")
        print(f"  {Style.GRAY}参数: {json.dumps(tc.function_args, ensure_ascii=False)}{Style.RESET}")
        print(f"  {Style.MAGENTA}原因: {info}{Style.RESET}")
    print_separator("━", 62, Style.RED)


def print_snapshot_info(snapshot: Snapshot, action: str):
    print(f"\n{Style.BLUE}{Style.BOLD}📸 快照 {action}{Style.RESET}")
    print(f"  {Style.GRAY}ID: {snapshot.id}{Style.RESET}")
    print(f"  {Style.GRAY}状态: {snapshot.state}{Style.RESET}")


# ================================================================
# Guard 审批 Schema
# ================================================================

class PaymentApprovalResponse(GuardPolicyHumanResponseSchema):
    """支付审批响应"""
    approved: bool = False
    reason: str = ""


class ItineraryModifyApprovalResponse(GuardPolicyHumanResponseSchema):
    """行程修改审批响应"""
    approved: bool = False
    reason: str = ""


# ================================================================
# 工具定义
# ================================================================

# ---- 普通工具 (1/3) ----

@tool_creator(
    tool_name="get_weather",
    tool_description="查询指定城市的天气信息。返回当前温度、天气状况和建议穿着。",
    labels=["travel", "weather"],
)
async def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_db = {
        "北京": {"temp": "22°C", "condition": "晴朗", "clothing": "薄外套 + T恤"},
        "上海": {"temp": "26°C", "condition": "多云", "clothing": "短袖 + 防晒"},
        "成都": {"temp": "20°C", "condition": "小雨", "clothing": "长袖 + 带伞"},
        "三亚": {"temp": "32°C", "condition": "晴朗", "clothing": "短袖短裤 + 防晒霜"},
        "哈尔滨": {"temp": "-5°C", "condition": "大雪", "clothing": "羽绒服 + 雪地靴"},
        "东京": {"temp": "18°C", "condition": "晴间多云", "clothing": "薄外套"},
        "巴黎": {"temp": "15°C", "condition": "阴天", "clothing": "风衣 + 围巾"},
    }
    info = weather_db.get(city)
    if info:
        return f"🌤 {city}天气: {info['condition']}, 温度: {info['temp']}, 建议穿着: {info['clothing']}"
    return f"🌤 {city}天气: 晴朗, 温度: 25°C, 建议穿着: 休闲装（默认信息）"


@tool_creator(
    tool_name="search_flights",
    tool_description="搜索从出发城市到目的城市的航班信息，返回可用航班列表。",
    labels=["travel", "flights"],
)
async def search_flights(departure_city: str, destination_city: str, date: str) -> str:
    """搜索航班"""
    flights = [
        {"flight": f"CA{hash(departure_city + destination_city) % 9000 + 1000}",
         "dep_time": "08:30", "arr_time": "11:45", "price": "¥1,280", "airline": "中国国航"},
        {"flight": f"MU{hash(destination_city + departure_city) % 9000 + 1000}",
         "dep_time": "14:00", "arr_time": "17:15", "price": "¥980", "airline": "东方航空"},
        {"flight": f"CZ{hash(departure_city + date) % 9000 + 1000}",
         "dep_time": "19:30", "arr_time": "22:45", "price": "¥1,560", "airline": "南方航空"},
    ]
    result = f"✈️ {departure_city} → {destination_city} ({date}) 航班:\n"
    for f in flights:
        result += f"  • {f['flight']} ({f['airline']}): {f['dep_time']}-{f['arr_time']}, {f['price']}\n"
    return result


# ---- Ask Human 工具 (1/3) ----

@tool_creator(
    tool_name="book_hotel",
    tool_description="预订指定城市的酒店。",
    labels=["travel", "hotel"],
    ask_human_policy=AskHumanPolicy(timeout=120),
)
async def book_hotel(city: str, check_in: str, check_out: str, tool_runtime: BaseToolRuntime = None) -> str:
    """预订酒店 - 需要向用户确认偏好"""
    try:
        response = await tool_runtime.ask_human(
            data={
                "question": f"正在为您预订 {city} 的酒店（{check_in} 至 {check_out}），请确认您的偏好：\n"
                            f"  1️⃣  标准间（约 ¥400/晚）\n"
                            f"  2️⃣  大床房（约 ¥550/晚）\n"
                            f"  3️⃣  豪华套房（约 ¥1,200/晚）\n"
                            f"请输入编号（1/2/3）或自定义需求："
            },
            timeout=120,
        )
        choice = response.get("answer", "2").strip()
        room_map = {
            "1": ("标准间", "¥400/晚"),
            "2": ("大床房", "¥550/晚"),
            "3": ("豪华套房", "¥1,200/晚"),
        }
        room_type, price = room_map.get(choice, ("大床房", "¥550/晚"))
        return (f"🏨 酒店预订成功！\n"
                f"  城市: {city}\n"
                f"  入住: {check_in} | 退房: {check_out}\n"
                f"  房型: {room_type} ({price})\n"
                f"  确认号: HTL-{hash(city + check_in) % 900000 + 100000}")
    except Exception as e:
        return f"🏨 酒店预订请求异常（{type(e).__name__}），已为您默认选择大床房 ¥550/晚。确认号: HTL-{hash(city) % 900000 + 100000}"


@tool_creator(
    tool_name="recommend_restaurant",
    tool_description="推荐指定城市的餐厅。推荐前需要了解用户的口味偏好。",
    labels=["travel", "food"],
    ask_human_policy=AskHumanPolicy(timeout=120),
)
async def recommend_restaurant(city: str, tool_runtime: BaseToolRuntime = None) -> str:
    """推荐餐厅 - 需要询问用户口味"""
    try:
        response = await tool_runtime.ask_human(
            data={
                "question": f"为您推荐 {city} 的餐厅，请告诉我您的口味偏好：\n"
                            f"  🌶 辣味  |  🍜 清淡  |  🍣 日料  |  🥩 西餐  |  🍲 火锅\n"
                            f"请输入您喜欢的口味（可多选，用空格分隔）："
            },
            timeout=120,
        )
        preference = response.get("answer", "清淡").strip()
        restaurants_db = {
            "辣味": [("川味坊", "⭐⭐⭐⭐⭐", "人均 ¥120"), ("辣妹子火锅", "⭐⭐⭐⭐", "人均 ¥150")],
            "清淡": [("江南小筑", "⭐⭐⭐⭐⭐", "人均 ¥180"), ("粤味轩", "⭐⭐⭐⭐", "人均 ¥200")],
            "日料": [("筑地寿司", "⭐⭐⭐⭐⭐", "人均 ¥300"), ("樱花亭", "⭐⭐⭐⭐", "人均 ¥250")],
            "西餐": [("米其林牛排馆", "⭐⭐⭐⭐⭐", "人均 ¥400"), ("巴黎小厨", "⭐⭐⭐⭐", "人均 ¥280")],
            "火锅": [("海底捞", "⭐⭐⭐⭐⭐", "人均 ¥160"), ("小龙坎", "⭐⭐⭐⭐", "人均 ¥130")],
        }
        result = f"🍽 为您推荐 {city} 的餐厅（偏好: {preference}）：\n"
        found = False
        for key, rests in restaurants_db.items():
            if key in preference:
                found = True
                for name, rating, price in rests:
                    result += f"  • {name} {rating} {price}\n"
        if not found:
            result += "  • 江南小筑 ⭐⭐⭐⭐⭐ 人均 ¥180\n"
            result += "  • 粤味轩 ⭐⭐⭐⭐ 人均 ¥200\n"
        return result
    except Exception as e:
        return f"🍽 餐厅推荐请求异常（{type(e).__name__}），为您推荐默认热门餐厅：江南小筑 ⭐⭐⭐⭐⭐ 人均 ¥180"


# ---- Guard 工具 (1/3) ----

def _payment_guard_func(response: PaymentApprovalResponse) -> bool:
    return getattr(response, "approved", False)

def _payment_reject_func(response: PaymentApprovalResponse) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="_placeholder_",
        name="process_payment",
        content=f"⛔ 支付已被用户拒绝。原因: {getattr(response, 'reason', '用户未批准')}",
        is_error=False,
    )

@tool_creator(
    tool_name="process_payment",
    tool_description="处理旅行相关的支付操作，包括机票、酒店等费用结算。涉及资金操作，需要用户确认。",
    labels=["travel", "payment"],
    guard_policy=GuardPolicy(
        info="💳 支付操作需要您的确认审批",
        schema=PaymentApprovalResponse(),
        guard_func=_payment_guard_func,
        reject_func=_payment_reject_func,
    ),
)
async def process_payment(item: str, amount: str, payment_method: str) -> str:
    """处理支付"""
    txn_id = f"TXN-{hash(item + amount) % 9000000 + 1000000}"
    return (f"💳 支付成功！\n"
            f"  项目: {item}\n"
            f"  金额: {amount}\n"
            f"  支付方式: {payment_method}\n"
            f"  交易号: {txn_id}")


def _itinerary_guard_func(response: ItineraryModifyApprovalResponse) -> bool:
    return getattr(response, "approved", False)

def _itinerary_reject_func(response: ItineraryModifyApprovalResponse) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="_placeholder_",
        name="modify_itinerary",
        content=f"⛔ 行程修改已被用户拒绝。原因: {getattr(response, 'reason', '用户未批准')}",
        is_error=False,
    )

@tool_creator(
    tool_name="modify_itinerary",
    tool_description="修改用户的旅行行程安排。行程变更影响较大，需要用户确认后才能执行。",
    labels=["travel", "itinerary"],
    guard_policy=GuardPolicy(
        info="📝 行程修改需要您的确认审批",
        schema=ItineraryModifyApprovalResponse(),
        guard_func=_itinerary_guard_func,
        reject_func=_itinerary_reject_func,
    ),
)
async def modify_itinerary(original_plan: str, modification: str, reason: str) -> str:
    """修改行程"""
    return (f"📝 行程已修改！\n"
            f"  原计划: {original_plan}\n"
            f"  修改为: {modification}\n"
            f"  修改原因: {reason}\n"
            f"  修改确认号: MOD-{hash(original_plan + modification) % 900000 + 100000}")


# ================================================================
# 自定义 ExecutingTools 生命周期（支持 AskHuman 交互）
# ================================================================

class DemoExecutingTools(DefaultExecutingTools):
    """
    演示用的工具执行生命周期处理器。

    在默认工具执行的基础上，启动后台任务监听事件通道中的 AskHumanEvent，
    实现在工具执行过程中与用户进行实时交互。
    """

    async def execute(self, data: ExecutingToolsRequest) -> ExecutingToolsResponse:
        consumer_task = asyncio.create_task(
            self._ask_human_consumer(data.event_channel)
        )
        try:
            return await super().execute(data)
        finally:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

    async def _ask_human_consumer(self, event_channel):
        """后台消费 AskHumanEvent 并与用户交互"""
        while True:
            try:
                event = await event_channel.receive_event(timeout=300)
                if hasattr(event, "type") and event.type == "ask_human_event":
                    content = event.data.content
                    question = content.get("question", "请输入您的选择：")

                    # 显示提问
                    print(f"\n{Style.MAGENTA}{Style.BOLD}🤖 工具提问:{Style.RESET}")
                    print(f"{Style.CYAN}{question}{Style.RESET}")

                    # 异步获取用户输入
                    user_answer = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input(f"{Style.GREEN}👤 你的回答: {Style.RESET}")
                    )

                    # 构建响应
                    response_event = AskHumanResponseEvent(
                        data=AskHumanResponseEvent.AskHumanResponseEventData(
                            response_success=True,
                            message="success",
                            response_content={"answer": user_answer},
                        )
                    )
                    # 设置 Future 结果
                    if not event.data.ask_human_response_channel.done():
                        event.data.ask_human_response_channel.set_result(response_event)
            except asyncio.CancelledError:
                break
            except Exception:
                break


# ================================================================
# Guard 审批交互
# ================================================================

async def handle_guard_triggered(
    agent: Agent,
    guard_event: GuardTriggeredEvent,
    stream_mode: str = "chunk",
):
    """
    处理 Guard 触发事件：
    1. 展示待审批的工具调用信息
    2. 序列化 Snapshot
    3. 收集用户审批
    4. 反序列化 Snapshot
    5. 恢复 Agent 执行
    """
    contexts = guard_event.data.guard_triggered_contexts
    snapshot = guard_event.data.snapshot

    print_guard_prompt(contexts)

    # ---- Snapshot 序列化演示 ----
    print_snapshot_info(snapshot, "序列化中...")
    serialized_data = snapshot.serialize()
    print(f"  {Style.GRAY}序列化大小: {len(serialized_data):,} bytes{Style.RESET}")
    print_success("快照已序列化为字节数据")

    # ---- 模拟持久化（写入/读取）----
    print(f"  {Style.GRAY}[模拟] 持久化存储 → snapshot_{snapshot.id}.bin{Style.RESET}")

    # ---- Snapshot 反序列化 ----
    restored_snapshot = Snapshot.deserialize(serialized_data)
    print_snapshot_info(restored_snapshot, "反序列化完成")
    print_success(f"快照已恢复 (ID: {restored_snapshot.id})")

    # ---- 收集人工审批 ----
    human_response: Dict[str, GuardPolicyHumanResponseSchema] = {}

    for ctx in contexts:
        tc = ctx.tool_call
        tool = ctx.tool_info
        print(f"\n  {Style.YELLOW}[{tc.function_name}]{Style.RESET} "
              f"参数: {json.dumps(tc.function_args, ensure_ascii=False)}")

        user_input = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: input(f"  {Style.GREEN}是否批准？(y/n): {Style.RESET}")
        )
        approved = user_input.strip().lower() in ("y", "yes", "是")

        reason = ""
        if not approved:
            reason = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input(f"  {Style.GRAY}拒绝原因（可选）: {Style.RESET}")
            )

        if tc.function_name == "process_payment":
            human_response[tc.tool_call_id] = PaymentApprovalResponse(
                approved=approved, reason=reason
            )
        elif tc.function_name == "modify_itinerary":
            human_response[tc.tool_call_id] = ItineraryModifyApprovalResponse(
                approved=approved, reason=reason
            )
        else:
            human_response[tc.tool_call_id] = GuardPolicyHumanResponseSchema()

        status = f"{Style.GREEN}✔ 已批准" if approved else f"{Style.RED}✘ 已拒绝"
        print(f"  {status}{Style.RESET}")

    print_separator("━", 62, Style.RED)

    # ---- 使用反序列化的 Snapshot 恢复执行 ----
    print_info("正在恢复 Agent 执行...")
    return restored_snapshot, human_response


# ================================================================
# 主事件处理循环
# ================================================================

async def process_agent_stream(agent, user_input_msg, stream_mode="chunk"):
    """处理 Agent 流式输出，返回是否需要继续（Guard 恢复场景）"""
    reasoning_started = False
    content_started = False

    async for event in agent.stream(user_input_msg, stream_mode=stream_mode):
        # ---- 流式 Chunk 事件 ----
        if isinstance(event, AssistantMessageChunkOutputEvent):
            chunk_data = event.data
            if chunk_data.chunk_type == "reasoning_content" and chunk_data.reasoning_content:
                if not reasoning_started:
                    reasoning_started = True
                    print(f"\n{Style.DIM}{Style.MAGENTA}💭 思考中: ", end="", flush=True)
                print(f"{chunk_data.reasoning_content}", end="", flush=True)

            elif chunk_data.chunk_type == "content" and chunk_data.content:
                if reasoning_started and not content_started:
                    print(f"{Style.RESET}")  # 结束思考行
                    reasoning_started = False
                if not content_started:
                    content_started = True
                    print(f"\n{Style.GREEN}{Style.BOLD}🤖 助手: {Style.RESET}{Style.GREEN}", end="", flush=True)
                print(f"{chunk_data.content}", end="", flush=True)

            elif chunk_data.chunk_type == "refusal" and chunk_data.refusal:
                print(f"\n{Style.RED}⚠ 拒绝: {chunk_data.refusal}{Style.RESET}", end="", flush=True)

        # ---- 完整 Assistant 消息事件 ----
        elif isinstance(event, AssistantMessageOutputEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")  # 结束当前行
            # 如果没有流式输出，且有 content
            if not content_started and event.data.content:
                print(f"\n{Style.GREEN}{Style.BOLD}🤖 助手: {Style.RESET}{Style.GREEN}{event.data.content}{Style.RESET}")

        # ---- 工具调用检测事件 ----
        elif isinstance(event, ToolCallEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
                content_started = False
                reasoning_started = False
            print_tool_header(event.data.function_name, event.data.function_args)

        # ---- 工具执行完毕事件 ----
        elif isinstance(event, ToolsExecutedEvent):
            print(f"\n{Style.YELLOW}{Style.BOLD}📦 工具执行完毕:{Style.RESET}")
            print_tool_result(event.data.tool_results)

        # ---- Guard 触发事件 ----
        elif isinstance(event, GuardTriggeredEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
                content_started = False
                reasoning_started = False

            restored_snapshot, human_response = await handle_guard_triggered(
                agent, event, stream_mode
            )

            # 使用恢复的 Snapshot 继续执行
            await process_agent_resume(agent, restored_snapshot, human_response, stream_mode)
            return

        # ---- 中断事件 ----
        elif isinstance(event, InterruptEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
            print_error(f"Agent 被中断: {event.data.reason}")

            # 演示 Snapshot 序列化
            if event.data.snapshot:
                snapshot = event.data.snapshot
                print_snapshot_info(snapshot, "中断快照")
                serialized = snapshot.serialize()
                print(f"  {Style.GRAY}序列化大小: {len(serialized):,} bytes{Style.RESET}")
                restored = Snapshot.deserialize(serialized)
                print_success(f"中断快照已序列化/反序列化 (ID: {restored.id})")
            return

        # ---- 轮次结束事件 ----
        elif isinstance(event, RoundStopEvent):
            pass

    if content_started or reasoning_started:
        print(f"{Style.RESET}")


async def process_agent_resume(agent, snapshot, human_response, stream_mode="chunk"):
    """处理 Agent 恢复执行流"""
    reasoning_started = False
    content_started = False

    async for event in agent.resume_stream(snapshot, human_response, stream_mode):
        if isinstance(event, AssistantMessageChunkOutputEvent):
            chunk_data = event.data
            if chunk_data.chunk_type == "reasoning_content" and chunk_data.reasoning_content:
                if not reasoning_started:
                    reasoning_started = True
                    print(f"\n{Style.DIM}{Style.MAGENTA}💭 思考中: ", end="", flush=True)
                print(f"{chunk_data.reasoning_content}", end="", flush=True)
            elif chunk_data.chunk_type == "content" and chunk_data.content:
                if reasoning_started and not content_started:
                    print(f"{Style.RESET}")
                    reasoning_started = False
                if not content_started:
                    content_started = True
                    print(f"\n{Style.GREEN}{Style.BOLD}🤖 助手: {Style.RESET}{Style.GREEN}", end="", flush=True)
                print(f"{chunk_data.content}", end="", flush=True)
            elif chunk_data.chunk_type == "refusal" and chunk_data.refusal:
                print(f"\n{Style.RED}⚠ 拒绝: {chunk_data.refusal}{Style.RESET}", end="", flush=True)

        elif isinstance(event, AssistantMessageOutputEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
            if not content_started and event.data.content:
                print(f"\n{Style.GREEN}{Style.BOLD}🤖 助手: {Style.RESET}{Style.GREEN}{event.data.content}{Style.RESET}")

        elif isinstance(event, ToolCallEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
                content_started = False
                reasoning_started = False
            print_tool_header(event.data.function_name, event.data.function_args)

        elif isinstance(event, ToolsExecutedEvent):
            print(f"\n{Style.YELLOW}{Style.BOLD}📦 工具执行完毕:{Style.RESET}")
            print_tool_result(event.data.tool_results)

        elif isinstance(event, GuardTriggeredEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
                content_started = False
                reasoning_started = False
            restored_snapshot, new_human_response = await handle_guard_triggered(
                agent, event, stream_mode
            )
            await process_agent_resume(agent, restored_snapshot, new_human_response, stream_mode)
            return

        elif isinstance(event, InterruptEvent):
            if content_started or reasoning_started:
                print(f"{Style.RESET}")
            print_error(f"Agent 被中断: {event.data.reason}")
            if event.data.snapshot:
                snap = event.data.snapshot
                serialized = snap.serialize()
                restored = Snapshot.deserialize(serialized)
                print_success(f"中断快照已序列化/反序列化 (ID: {restored.id})")
            return

        elif isinstance(event, RoundStopEvent):
            pass

    if content_started or reasoning_started:
        print(f"{Style.RESET}")


# ================================================================
# 主入口
# ================================================================

async def main():
    print_banner()

    # ---- 配置 LLM ----
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print_error("请设置环境变量 DEEPSEEK_API_KEY 或在 examples/.env 文件中配置")
        print(f"  {Style.GRAY}export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx{Style.RESET}")
        print(f"  {Style.GRAY}或复制 examples/.env.example 为 examples/.env 并填写{Style.RESET}")
        return

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    llm_config = LLMConfig(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        provider="deepseek",
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
    )

    # ---- 构建上下文 ----
    tools = [get_weather, search_flights, book_hotel, recommend_restaurant, process_payment, modify_itinerary]

    context = Context(
        system_prompt=(
            "你是一位专业的智能旅行规划助手「旅行小管家」。\n\n"
            "你的能力包括：\n"
            "1. 查询目的地天气信息（get_weather）\n"
            "2. 搜索航班（search_flights）\n"
            "3. 预订酒店（book_hotel）- 预订前需确认用户偏好\n"
            "4. 推荐餐厅（recommend_restaurant）- 推荐前需了解用户口味\n"
            "5. 处理支付（process_payment）- 涉及资金操作需用户审批\n"
            "6. 修改行程（modify_itinerary）- 重要变更需用户确认\n\n"
            "请根据用户需求主动使用工具来帮助规划旅行。回答时注意：\n"
            "- 如果用户提到目的地，主动查询天气和搜索航班\n"
            "- 当需要预订酒店或推荐餐厅时，使用对应工具获取用户偏好\n"
            "- 当涉及支付或行程变更时，使用对应工具并等待用户审批\n"
            "- 回答要简洁友好，适当使用 emoji 增加趣味性\n"
            "- 使用中文回答"
        ),
        tool_manager=tools,
    )

    # ---- 构建 Agent ----
    lifespan_manager = LifespanManager(
        executing_tools=DemoExecutingTools(),
    )
    agent = Agent(
        llm_config=llm_config,
        context=context,
        lifespan_manager=lifespan_manager,
    )

    # ---- 工具汇总展示 ----
    print(f"{Style.BOLD}📋 已注册工具:{Style.RESET}")
    print_separator()
    tool_categories = {
        "普通工具": ["get_weather", "search_flights"],
        "Ask Human 工具": ["book_hotel", "recommend_restaurant"],
        "Guard 工具": ["process_payment", "modify_itinerary"],
    }
    icons = {"普通工具": "🔨", "Ask Human 工具": "🗣️", "Guard 工具": "🛡️"}
    for category, names in tool_categories.items():
        print(f"  {icons[category]} {Style.BOLD}{category}{Style.RESET}: {', '.join(names)}")
    print_separator()
    print(f"\n{Style.DIM}输入 quit 或 exit 退出对话{Style.RESET}\n")

    # ---- 对话主循环 ----
    round_count = 0
    while True:
        try:
            user_text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(f"{Style.BLUE}{Style.BOLD}👤 你: {Style.RESET}")
            )
        except (EOFError, KeyboardInterrupt):
            break

        user_text = user_text.strip()
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            print(f"\n{Style.CYAN}👋 感谢使用智能旅行规划助手，祝您旅途愉快！{Style.RESET}")
            break

        round_count += 1
        print_separator("─", 62, Style.GRAY)
        print(f"{Style.DIM}[第 {round_count} 轮对话]{Style.RESET}")

        user_msg = UserMessage(content=user_text)

        try:
            await process_agent_stream(agent, user_msg, stream_mode="chunk")
        except Exception as e:
            print_error(f"运行异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        print()
        print_separator("─", 62, Style.GRAY)


if __name__ == "__main__":
    asyncio.run(main())
