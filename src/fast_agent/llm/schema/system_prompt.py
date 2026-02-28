from typing import Union, List, Optional, Literal, Dict, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

DEFAULT_SYSTEM_PROMPT_CHIP_KEY = "default"

class SystemPromptChipMetadataSchema(BaseModel):
    """
    SystemPromptChipMetadataSchema 系统提示词切片元数据类，用于定义系统提示词切片的元数据信息，允许拓展字段
    """
    ignore: bool = False

    model_config = ConfigDict(extra="allow")

    @classmethod
    def default(cls) -> "SystemPromptChipMetadataSchema":
        """
        default 方法用于获取默认的 SystemPromptChipMetadataSchema 实例
        """
        return cls(ignore=False)



class SystemPromptChipSchema(BaseModel):
    """
    SystemPromptChipSchema 系统提示词具体切片类，用于定义系统提示词的具体切片结构
    """
    name: str

    content: str

    metadata: SystemPromptChipMetadataSchema = Field(default_factory=SystemPromptChipMetadataSchema.default)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def default(cls, content: str) -> "SystemPromptChipSchema":
        """
        default 方法用于获取默认的 SystemPromptChipSchema 实例
        """
        return cls(name=DEFAULT_SYSTEM_PROMPT_CHIP_KEY, content=content)



class SystemPromptChipsSchema(BaseModel):
    """
    SystemPromptChipsSchema 系统提示词切片集合类，用于定义系统提示词的切片集合结构
    """
    order: List[str] = Field(default_factory=list)
    splitter: str = "\n\n"
    chips: Dict[str, SystemPromptChipSchema] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def default(cls, content: str) -> "SystemPromptChipsSchema":
        """
        default 方法用于获取默认的 SystemPromptChipsSchema 实例
        """
        return cls(
            order=[DEFAULT_SYSTEM_PROMPT_CHIP_KEY],
            splitter="\n\n",
            chips={
                DEFAULT_SYSTEM_PROMPT_CHIP_KEY: SystemPromptChipSchema.default(content)
            },
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemPromptChipsSchema":
        """
        从 dict 构建，示例结构如下：

        {
            "order": [...],
            "splitter": "...",
            "chips": {"chip_key": {...}}
        }
        """
        order = data.get("order", [])
        splitter = data.get("splitter", "\n\n")

        chips_payload = data.get("chips")
        if not isinstance(chips_payload, dict):
            raise ValueError(
                "Error! Unsupported chips schema. Expected dict with key 'chips', e.g. {'order': [...], 'splitter': '...', 'chips': {...}}."
            )

        chips: Dict[str, SystemPromptChipSchema] = {}
        for key, value in chips_payload.items():
            chips[key] = cls._normalize_chip(key, value)

        return cls(order=order, splitter=splitter, chips=chips)

    @staticmethod
    def _normalize_chip(key: str, value: Any) -> SystemPromptChipSchema:
        if isinstance(value, SystemPromptChipSchema):
            chip = value
        elif isinstance(value, dict):
            payload = dict(value)
            payload.setdefault("name", key)
            chip = SystemPromptChipSchema(**payload)
        else:
            raise ValueError(
                f"Error! Unsupported chip type for key '{key}'. Expected dict or SystemPromptChipSchema."
            )

        if chip.name != key:
            chip = chip.model_copy(update={"name": key})

        return chip
    
    def to_str(self) -> str:
        """转换为 str 类型"""
        parts = []
        for key in self.order:
            chip = self.chips.get(key)
            if chip and not chip.metadata.ignore:
                parts.append(chip.content)

        return self.splitter.join(parts)

    def to_xml(self) -> str:
        """转换为 xml 类型"""
        parts = []
        for key in self.order:
            chip = self.chips.get(key)
            if chip and not chip.metadata.ignore:
                content = chip.content
                parts.append(f"<{key}>\n{content}\n</{key}>")

        return "\n".join(parts)
    

class SystemPrompt:
    """
    SystemPrompt 系统提示词类，用于定义与管理系统提示词

    Example:
    ```python

    chips = {
        "order": ["chip_key_1", "chip_key_2"], # chip 的顺序列表，若为空则表示不指定顺序
        "splitter": "\\n\\n", # chip 内容拼接时使用的分隔符，默认使用双换行符
        "chips": {
            "chip_key_1": {
                "name": "chip_key_1", # chip 的名称，必须与 key 保持一致
                "content": "...", # chip 内容
                "metadata": {
                    "ignore": False # 是否忽略该 chip 的拼接，默认为 False
                }
            },
            "chip_key_2": {
                "name": "chip_key_2",
                "content": "...",
                "metadata": {
                    "ignore": False
                }
            }
        }
    }
    ```
    """

    class PromptType(Enum):
        """PromptType 枚举类，用于定义获取的系统提示词表现类型"""
        STR = "str"
        XML = "xml"

    
    def __init__(
        self,
        content: Union[str, SystemPromptChipsSchema, dict],
        to_str: Optional[Callable[[SystemPromptChipsSchema], str]] = None,
        to_xml: Optional[Callable[[SystemPromptChipsSchema], str]] = None,
    ):
        if isinstance(content, str):
            self.chips = SystemPromptChipsSchema.default(content)

        elif isinstance(content, dict):
            self.chips = SystemPromptChipsSchema.from_dict(content)

        elif isinstance(content, SystemPromptChipsSchema):
            self.chips = content

        else:
            raise ValueError("Error! Unsupported content type for SystemPrompt. Expected str, dict, or SystemPromptChipsSchema.")
        
        # 自定义转换函数，优先级高于默认实现
        self.to_str = to_str
        self.to_xml = to_xml
        

    def get_system_prompt(self, type: Optional[Union[PromptType, Literal["str", "xml"]]] = None) -> str:
        """
        获取特定表现形式的系统提示词（默认为 str）
        """
        if type is None or type == self.PromptType.STR or type == "str":
            if self.to_str:
                return self.to_str(self.chips)
            return self.chips.to_str()
        
        elif type == self.PromptType.XML or type == "xml":
            if self.to_xml:
                return self.to_xml(self.chips)
            return self.chips.to_xml()
        
        else:
            raise ValueError("Error! Unsupported type for SystemPrompt. Expected str or xml.")
        
    
    # =================
    # chips API
    # =================

    def add(self, key: str, content: Union[str, SystemPromptChipSchema, dict]) -> SystemPromptChipSchema:
        """
        新增一个 chip。

        - 若 key 已存在则抛出异常
        - 新增成功后会自动追加到 order（若尚未存在）
        """
        if key in self.chips.chips:
            raise ValueError(f"Error! Chip '{key}' already exists.")

        if isinstance(content, str):
            chip = SystemPromptChipSchema(name=key, content=content)
        else:
            chip = SystemPromptChipsSchema._normalize_chip(key, content)

        self.chips.chips[key] = chip
        if key not in self.chips.order:
            self.chips.order.append(key)

        return chip

    def insert(self, key: str, content: Union[str, SystemPromptChipSchema, dict], index: int) -> SystemPromptChipSchema:
        """
        插入新增一个 chip 到指定 order 位置。

        - 若 key 已存在则抛出异常
        - index 必须在 [0, len(order)] 区间内
        """
        if key in self.chips.chips:
            raise ValueError(f"Error! Chip '{key}' already exists.")

        if index < 0 or index > len(self.chips.order):
            raise ValueError(
                f"Error! Invalid index '{index}'. Expected index in [0, {len(self.chips.order)}]."
            )

        if isinstance(content, str):
            chip = SystemPromptChipSchema(name=key, content=content)
        else:
            chip = SystemPromptChipsSchema._normalize_chip(key, content)

        self.chips.chips[key] = chip
        self.chips.order.insert(index, key)
        return chip

    def move(self, key: str, index: int) -> None:
        """
        调整指定 chip 在 order 中的位置。
        """
        if key not in self.chips.chips:
            raise ValueError(f"Error! Chip '{key}' does not exist.")

        if key not in self.chips.order:
            raise ValueError(f"Error! Chip '{key}' is not in order list.")

        if index < 0 or index >= len(self.chips.order):
            raise ValueError(
                f"Error! Invalid index '{index}'. Expected index in [0, {len(self.chips.order) - 1}]."
            )

        self.chips.order.remove(key)
        self.chips.order.insert(index, key)

    def remove(self, key: str) -> bool:
        """
        删除指定 key 的 chip。

        - 删除成功返回 True
        - key 不存在返回 False
        - 会同步从 order 中移除该 key
        """
        if key not in self.chips.chips:
            return False

        self.chips.chips.pop(key)
        self.chips.order = [chip_key for chip_key in self.chips.order if chip_key != key]
        return True

    def ignore(self, key: str) -> str:
        """
        软删除：将指定 chip 的 metadata.ignore 置为 True，并返回 key。
        """
        chip = self.get(key)
        if chip is None:
            raise ValueError(f"Error! Chip '{key}' does not exist.")

        chip.metadata.ignore = True
        return key

    def wakeup(self, key: str) -> str:
        """
        唤醒指定 chip：将 metadata.ignore 从 True 置为 False；若本来就是 False 则保持不变。
        """
        chip = self.get(key)
        if chip is None:
            raise ValueError(f"Error! Chip '{key}' does not exist.")

        chip.metadata.ignore = False
        return key

    def wakeup_all(self) -> List[str]:
        """
        唤醒全部 chip，返回本次被唤醒的 key 列表。
        """
        waked_up_keys: List[str] = []
        for key in self.chips.order:
            chip = self.chips.chips.get(key)
            if chip is not None and chip.metadata.ignore:
                chip.metadata.ignore = False
                waked_up_keys.append(key)

        return waked_up_keys

    def toggle(self, key: str) -> str:
        """
        切换指定 chip 的 metadata.ignore 状态，并返回 key。
        """
        chip = self.get(key)
        if chip is None:
            raise ValueError(f"Error! Chip '{key}' does not exist.")

        chip.metadata.ignore = not chip.metadata.ignore
        return key

    def replace_chips(self, content: Union[str, SystemPromptChipsSchema, dict]) -> None:
        """
        直接整体替换 chips。
        """
        if isinstance(content, str):
            self.chips = SystemPromptChipsSchema.default(content)
        elif isinstance(content, dict):
            self.chips = SystemPromptChipsSchema.from_dict(content)
        elif isinstance(content, SystemPromptChipsSchema):
            self.chips = content
        else:
            raise ValueError(
                "Error! Unsupported content type for replace_chips. Expected str, dict, or SystemPromptChipsSchema."
            )

    def update(self, content: Union[str, SystemPromptChipSchema, dict], key: str = DEFAULT_SYSTEM_PROMPT_CHIP_KEY) -> None:
        """
        更新系统提示词内容，根据指定的 key 和 content 更新对应的 chip 内容，content 允许为 str、SystemPromptChipSchema 或 dict 类型

        更新规则如下：
        1. 若不显式指定 key，则默认更新 DEFAULT_SYSTEM_PROMPT_CHIP_KEY 对应的 chip 内容
        2. 若指定的 key 不存在，则创建一个新的 chip，并将其添加到 chips 中，且将 key 添加到 order 列表末尾
        """
        if isinstance(content, str):
            chip = SystemPromptChipSchema(name=key, content=content)
        else:
            chip = SystemPromptChipsSchema._normalize_chip(key, content)

        self.chips.chips[key] = chip
        if key not in self.chips.order:
            self.chips.order.append(key)

    def get(self, key: str) -> Optional[SystemPromptChipSchema]:
        """
        获取指定 key 的 chip，不存在时返回 None。
        """
        return self.chips.chips.get(key)

    def get_chips(self) -> SystemPromptChipsSchema:
        """
        获取完整 chips 的深拷贝，避免外部直接修改内部状态。
        """
        return self.chips.model_copy(deep=True)


    
