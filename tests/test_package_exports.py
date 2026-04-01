import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_top_level_exports():
    from fast_agent import (
        Agent,
        Context,
        GuardPolicy,
        GuardPolicyHumanResponseSchema,
        LLMConfig,
        LifespanManager,
        Snapshot,
        UserMessage,
        __version__,
        tool_creator,
    )

    assert Agent.__name__ == "Agent"
    assert Context.__name__ == "Context"
    assert LLMConfig.__name__ == "LLMConfig"
    assert LifespanManager.__name__ == "LifespanManager"
    assert Snapshot.__name__ == "Snapshot"
    assert UserMessage.__name__ == "UserMessage"
    assert GuardPolicy.__name__ == "GuardPolicy"
    assert GuardPolicyHumanResponseSchema.__name__ == "GuardPolicyHumanResponseSchema"
    assert callable(tool_creator)
    assert __version__ == "0.2.2"


def test_subpackage_exports():
    from fast_agent.resources import OpenSandboxFactory, ToolManager
    from fast_agent.types import CommandOpts, ISandBox, ToolResultMessage

    assert OpenSandboxFactory.__name__ == "OpenSandboxFactory"
    assert ToolManager.__name__ == "ToolManager"
    assert CommandOpts.__name__ == "CommandOpts"
    assert ISandBox.__name__ == "ISandBox"
    assert ToolResultMessage.__name__ == "ToolResultMessage"