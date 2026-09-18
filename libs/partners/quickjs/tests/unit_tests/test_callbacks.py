"""Callback propagation tests for QuickJS bridge dispatches."""

from __future__ import annotations

from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves annotations at runtime
)
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeChatModel(GenericFakeChatModel):
    """Generic fake chat model that supports tool binding."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> _FakeChatModel:
        del tools
        return self


class _ModelStartRecorder(BaseCallbackHandler):
    """Count model runs visible through the outer agent callbacks."""

    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        del serialized, messages, kwargs
        self.count += 1


async def test_task_global_propagates_outer_callbacks_to_subagent() -> None:
    """A subagent launched through JavaScript remains visible to outer callbacks."""
    main_model = _FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_eval",
                            "name": "eval",
                            "args": {
                                "code": (
                                    "await task({description: 'work', "
                                    "subagentType: 'worker'})"
                                )
                            },
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    worker_model = _FakeChatModel(messages=iter([AIMessage(content="ok")]))
    agent = create_deep_agent(
        model=main_model,
        middleware=[CodeInterpreterMiddleware()],
        subagents=[
            {
                "name": "worker",
                "description": "Does one unit of work.",
                "system_prompt": "Reply with ok.",
                "model": worker_model,
            }
        ],
    )
    recorder = _ModelStartRecorder()

    await agent.ainvoke(
        {"messages": [HumanMessage(content="Use eval to ask the worker.")]},
        config={"callbacks": [recorder]},
    )

    assert recorder.count == 3
