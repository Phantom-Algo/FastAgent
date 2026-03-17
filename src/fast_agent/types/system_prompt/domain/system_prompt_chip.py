from ...constant import DEFAULT_SYSTEM_PROMPT_CHIP_KEY
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Any


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