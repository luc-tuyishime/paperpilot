"""Agent tests with a scripted model: we decide in advance which tools it "chooses".
This tests OUR loop (routing, tool execution, memory, step limit), not the LLM's judgement."""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from app.api.routes.agent import get_chat_model, get_checkpointer

GAN = "A generator is trained against a discriminator in a minimax game. " * 6


class ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order; records every input it receives."""

    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    repeat_last: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:  # noqa: E501
        self.seen.append(list(messages))
        if self.repeat_last and len(self.script) == 1:
            # a NEW message each time, like a real model (same id = LangGraph updates in place)
            base, n = self.script[0], len(self.seen)
            calls = [{**c, "id": f"{c['id']}-{n}"} for c in base.tool_calls]
            message = AIMessage(content=base.content, tool_calls=calls)
        else:
            message = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kw: Any) -> "ScriptedModel":  # type: ignore[override]
        return self


def call(tool: str, args: dict[str, Any], id_: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": id_}])


@pytest.fixture
def agent_client(client: TestClient, make_pdf: Callable[..., bytes]) -> TestClient:
    client.post(
        "/documents", files={"file": ("gan.pdf", make_pdf(["intro", GAN]), "application/pdf")}
    )
    saver = InMemorySaver()  # memory in RAM for tests; PostgreSQL in the app
    client.app.dependency_overrides[get_checkpointer] = lambda: saver  # type: ignore[attr-defined]
    return client


def use(client: TestClient, model: ScriptedModel) -> None:
    client.app.dependency_overrides[get_chat_model] = lambda: model  # type: ignore[attr-defined]


def test_agent_searches_then_answers_with_a_trace(agent_client: TestClient) -> None:
    model = ScriptedModel(
        script=[
            call("search_papers", {"query": "how is the generator trained"}, "c1"),
            AIMessage(content="G is trained against D in a minimax game (A Test Paper, p. 2)."),
        ]
    )
    use(agent_client, model)

    body = agent_client.post("/agent", json={"message": "How is the generator trained?"}).json()

    assert body["answer"].startswith("G is trained")
    assert [s["tool"] for s in body["steps"]] == ["search_papers"]
    assert "p. 2" in body["steps"][0]["result_preview"]  # the tool really searched Qdrant
    assert body["thread_id"] and body["stopped_early"] is False
    # the second model call saw the tool result
    assert any(isinstance(m, ToolMessage) and "minimax" in m.text for m in model.seen[1])


def test_same_thread_remembers_the_conversation(agent_client: TestClient) -> None:
    model = ScriptedModel(script=[AIMessage(content="First answer."), AIMessage(content="Second.")])
    use(agent_client, model)

    first = agent_client.post("/agent", json={"message": "Tell me about GANs"}).json()
    agent_client.post(
        "/agent", json={"message": "And the discriminator?", "thread_id": first["thread_id"]}
    )

    humans = [m.text for m in model.seen[1] if isinstance(m, HumanMessage)]
    assert humans == ["Tell me about GANs", "And the discriminator?"]  # memory works
    assert any(m.text == "First answer." for m in model.seen[1])


def test_new_thread_starts_empty(agent_client: TestClient) -> None:
    model = ScriptedModel(script=[AIMessage(content="One."), AIMessage(content="Two.")])
    use(agent_client, model)

    agent_client.post("/agent", json={"message": "Tell me about GANs"})
    agent_client.post("/agent", json={"message": "Something else"})  # no thread_id

    humans = [m.text for m in model.seen[1] if isinstance(m, HumanMessage)]
    assert humans == ["Something else"]


def test_step_limit_stops_a_looping_agent(agent_client: TestClient) -> None:
    model = ScriptedModel(script=[call("list_papers", {}, "loop")], repeat_last=True)
    use(agent_client, model)

    body = agent_client.post("/agent", json={"message": "Loop forever please"}).json()

    assert body["stopped_early"] is True
    assert body["answer"].startswith("Stopped after")
    assert len(model.seen) == 6  # exactly agent_max_steps model calls, then stop


def test_unknown_tool_is_reported_to_the_model_not_crashed(agent_client: TestClient) -> None:
    model = ScriptedModel(
        script=[call("delete_everything", {}, "x"), AIMessage(content="I can't do that.")]
    )
    use(agent_client, model)

    body = agent_client.post("/agent", json={"message": "Delete all papers"}).json()

    assert body["answer"] == "I can't do that."
    assert body["steps"][0]["result_preview"].startswith("Unknown tool")


def test_list_papers_shows_ready_documents(agent_client: TestClient) -> None:
    model = ScriptedModel(script=[call("list_papers", {}, "l1"), AIMessage(content="One paper.")])
    use(agent_client, model)

    body = agent_client.post("/agent", json={"message": "What papers do you have?"}).json()

    assert "A Test Paper" in body["steps"][0]["result_preview"]
