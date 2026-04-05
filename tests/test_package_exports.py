import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_top_level_exports():
    from fast_agent import (
        Agent,
        Context,
        EmbeddingConfig,
        EmbeddingResponse,
        GuardPolicy,
        GuardPolicyHumanResponseSchema,
        LLMConfig,
        LifespanManager,
        MessageRound,
        Snapshot,
        UserMessage,
        __version__,
        tool_creator,
    )

    assert Agent.__name__ == "Agent"
    assert Context.__name__ == "Context"
    assert EmbeddingConfig.__name__ == "EmbeddingConfig"
    assert EmbeddingResponse.__name__ == "EmbeddingResponse"
    assert LLMConfig.__name__ == "LLMConfig"
    assert LifespanManager.__name__ == "LifespanManager"
    assert MessageRound.__name__ == "MessageRound"
    assert Snapshot.__name__ == "Snapshot"
    assert UserMessage.__name__ == "UserMessage"
    assert GuardPolicy.__name__ == "GuardPolicy"
    assert GuardPolicyHumanResponseSchema.__name__ == "GuardPolicyHumanResponseSchema"
    assert callable(tool_creator)
    assert __version__ == "0.2.2"


def test_subpackage_exports():
    from fast_agent.resources import EmbeddingConfig, OpenSandboxFactory, ToolManager
    from fast_agent.types import BaseEmbeddingConfig, CommandOpts, EmbeddingVector, ISandBox, MessageRound, ToolResultMessage

    assert EmbeddingConfig.__name__ == "EmbeddingConfig"
    assert OpenSandboxFactory.__name__ == "OpenSandboxFactory"
    assert ToolManager.__name__ == "ToolManager"
    assert BaseEmbeddingConfig.__name__ == "BaseEmbeddingConfig"
    assert CommandOpts.__name__ == "CommandOpts"
    assert EmbeddingVector.__name__ == "EmbeddingVector"
    assert ISandBox.__name__ == "ISandBox"
    assert MessageRound.__name__ == "MessageRound"
    assert ToolResultMessage.__name__ == "ToolResultMessage"