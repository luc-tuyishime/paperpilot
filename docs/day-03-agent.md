# Day 3 — The agent

RAG (`/ask`) does **one** search, then answers. The agent (`/agent`) **decides** what to do:
it can search several times, for example once per paper, before answering. And it remembers
the conversation.

```
START → agent ──(asks for a tool?)── yes → tools → back to agent
                                  └─ no  → END: final answer
```

- **agent**: Gemini reads the conversation and either asks for a tool or answers.
- **tools**: our code runs the tool and adds its result to the conversation.
- The loop stops when the model answers, or after **6 model calls** (the guardrail).

## Run it (30 min)

```bash
uv sync                                     # new packages: langgraph, langchain
uv run pytest -q                            # 40 passed
uv run uvicorn app.main:app --reload        # restart
```

Ask something that needs two searches:

```bash
curl -s localhost:8000/agent -H "Content-Type: application/json" \
  -d '{"message": "Compare how attention is used in Bahdanau et al. and in the Transformer."}' | python3 -m json.tool
```

Look at the response:
- `steps`: every tool the agent called, with its arguments. Expect two `search_papers`, one per paper.
- `answer`: the comparison, citing (Title, p. N).
- `thread_id`: the conversation's id.

**Memory.** Send a follow-up with that `thread_id`:

```bash
curl -s localhost:8000/agent -H "Content-Type: application/json" \
  -d '{"message": "Which one came first?", "thread_id": "PASTE-IT-HERE"}' | python3 -m json.tool
```

It knows "which one" means the two papers. In Postico you'll see 4 new tables, `checkpoints`
and friends, which LangGraph creates to save each conversation.

## Read (in this order)

1. `app/agent/tools.py`: two tools. The **docstrings are instructions to the model**: it reads them to decide when to call each tool.
2. `app/agent/graph.py`: the graph (2 nodes, 1 loop), the step limit, and how the trace is collected.
3. `app/agent/runtime.py`: the chat model (LangChain handles tool calling for each provider) and the PostgreSQL memory.
4. `tests/test_agent.py`: a scripted fake model "chooses" tools, so we test our loop, not the LLM.

## Interview talking points

1. **RAG vs agent:** RAG is a fixed pipeline, one search then answer: fast, cheap, predictable. An agent decides its own steps: better for comparisons and multi-step questions, slower and costlier. Use the simplest one that works.
2. **Memory:** LangGraph saves the conversation state in PostgreSQL after every step, under a thread id. Same id, same conversation. It survives a server restart.
3. **Guardrails:** a step limit (6 model calls), only 2 read-only tools, tool errors returned to the model instead of crashing, and a trace of every tool call.
4. **Why LangGraph here but not for RAG:** the loop, the state and the checkpointing are what LangGraph is built for. The RAG core stays plain code, where every step is visible.

Then commit: `git add . && git commit -m "Day 3: LangGraph agent with tools and memory" && git push`
