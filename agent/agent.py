import argparse
import importlib.util
import json
import logging
import re as _re
import sys
from pathlib import Path
from typing import Any

import tiktoken

logger = logging.getLogger("logger")
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer

# Tool schema for memory retrieval — description will be refined in prompt work.
class _MemoryItem:
    """Local MemoryItem that avoids qdrant_client/fastembed deps in the outer memorybank."""

    def __init__(self, title: str = "", context: str = "", content: str = "", polarity: Any = None) -> None:
        self.title = title
        self.context = context
        self.content = content
        self.polarity = polarity

    @staticmethod
    def from_string(memory_str: str) -> list["_MemoryItem"]:
        tag_match = _re.search(r"<extracted_memories>(.*?)</extracted_memories>", memory_str, _re.DOTALL)
        parse_str = tag_match.group(1) if tag_match else memory_str
        parse_str = parse_str.replace("Memory Item", "# Memory Item")
        items: list[_MemoryItem] = []
        for chunk in parse_str.split("# Memory Item"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                lines = [line.strip() for line in chunk.splitlines() if line.strip()]
                if not lines:
                    continue
                int(lines[0].split()[0])  # first line must be a numeric index
                title = context = content = ""
                current_field: str | None = None
                for line in lines[1:]:
                    ll = line.lower()
                    if ll.startswith("## title"):
                        current_field = "title"
                        title = line[len("## title"):].strip()
                    elif ll.startswith("## context"):
                        current_field = "context"
                        context = line[len("## context"):].strip()
                    elif ll.startswith("## content"):
                        current_field = "content"
                        content = line[len("## content"):].strip()
                    elif ll.startswith("## "):
                        current_field = None
                    elif current_field == "title":
                        title = (title + " " + line).strip()
                    elif current_field == "context":
                        context = (context + " " + line).strip()
                    elif current_field == "content":
                        content = (content + " " + line).strip()
                items.append(_MemoryItem(title=title, context=context, content=content))
            except Exception:
                continue
        return items


_RETRIEVE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_memory",
        "description": (
            "The memory retriever retrieves transferable reasoning strategies, "
            "decision guidelines, failure-recovery lessons, and failure-prevention "
            "lessons distilled from prior task attempts for the current task state "
            "or subtask. Use it when planning or replanning a task or subtask, "
            "before committing to an important action, or when progress is uncertain, "
            "to get guidance on action selection and next-step reasoning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A good query should specify: "
                        "(1) the current task or subgoal, "
                        "(2) the current environment state, relevant constraints, "
                        "relevant observations from previous steps, and actions "
                        "already tried for this subtask, "
                        "(3) the decision, uncertainty, or failure mode you need "
                        "help with, and "
                        "(4) the kind of guidance you want, such as planning, search, "
                        "recovery from failures, or failure prevention. "
                        "Generate queries that ask for reusable reasoning and planning "
                        "guidance based in the current task state, rather than asking "
                        "for memories for task-specific facts or final answers."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        memory_client: Any = None,
        extraction_lm_config: Any = None,
        top_k: int = 3,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.memory_client = memory_client
        self.extraction_lm_config = extraction_lm_config
        self.top_k = top_k
        self._memory_trace: list[dict] = []

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        prompt = self.prompt_constructor.construct(trajectory, intent, meta_data)
        lm_config = self.lm_config
        force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")

        step = (len(trajectory) - 1) // 2

        if self.memory_client is not None:
            # Tool-calling path: retrieval happens in-step via the API tool loop.
            tools = [_RETRIEVE_MEMORY_TOOL]
            messages: list[dict[str, Any]] = list(prompt)  # mutable copy
            n = 0
            while True:
                msg = call_llm(lm_config, messages, tools=tools)

                if msg.tool_calls:
                    # Append the assistant turn, then execute and append each result.
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    })
                    for tc in msg.tool_calls:
                        query = json.loads(tc.function.arguments).get("query", "")
                        logger.info(f"[MEMORY_CALL] query={query[:300]!r}")
                        result = self.memory_client.retrieve(query, self.top_k)
                        logger.info(f"[MEMORY_RESULT] returned {len(result)} chars")
                        self._memory_trace.append({"step": step, "query": query, "retrieved": result})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    # Remind the model to output in the required action format, not tool-call XML.
                    messages.append({
                        "role": "user",
                        "content": (
                            "Memory retrieved. Now output your next browser action — "
                            "do NOT call any tools. End your response with:\n"
                            "In summary, the next action I will perform is "
                            "```action [args]```"
                        ),
                    })
                    # At most one retrieval per step — call without tools to force a browser action.
                    response = f"{force_prefix}{call_llm(lm_config, messages) or ''}"
                else:
                    response = f"{force_prefix}{msg.content or ''}"

                n += 1
                try:
                    parsed_response = self.prompt_constructor.extract_action(response)
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    else:
                        raise ValueError(f"Unknown action type {self.action_set_tag}")
                    action["raw_prediction"] = response
                    break
                except ActionParsingError:
                    if n >= lm_config.gen_config["max_retry"]:
                        action = create_none_action()
                        action["raw_prediction"] = response
                        break
                    # Retry: call again with the same messages (no bad turn appended).
        else:
            # Baseline path: no tools, plain text response.
            n = 0
            while True:
                response = call_llm(lm_config, prompt)
                response = f"{force_prefix}{response}"
                n += 1
                try:
                    parsed_response = self.prompt_constructor.extract_action(response)
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    else:
                        raise ValueError(f"Unknown action type {self.action_set_tag}")
                    action["raw_prediction"] = response
                    break
                except ActionParsingError:
                    if n >= lm_config.gen_config["max_retry"]:
                        action = create_none_action()
                        action["raw_prediction"] = response
                        break

        return action

    @property
    def memory_trace(self) -> list[dict]:
        """Per-task list of {"query": str, "retrieved": str} for each retrieve_memory call."""
        return list(self._memory_trace)

    def reset(self, test_config_file: str) -> None:
        self._memory_trace = []

    def extract_and_save_memories(
        self, trajectory: Trajectory, intent: str, score: float
    ) -> None:
        if self.memory_client is None or self.extraction_lm_config is None:
            return

        # Load models/prompts.py directly to bypass models/__init__.py, which pulls
        # in fastembed/qdrant not installed in the webarena conda environment.
        _mb_root = str(Path(__file__).parent.parent.parent / "memorybank")
        try:
            _spec = importlib.util.spec_from_file_location(
                "_mb_models_prompts", f"{_mb_root}/models/prompts.py"
            )
            _pmod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_pmod)
            webarena_prompts = _pmod.webarena_prompts
        except Exception as e:
            logger.warning(f"[MemoryExtraction] Cannot load webarena_prompts: {e}")
            return

        prompt_key = "success_extraction" if score == 1 else "failure_extraction"
        extraction_prompt = webarena_prompts[prompt_key]

        # build a text summary of the trajectory
        traj_lines = [f"OBJECTIVE: {intent}"]
        for i, item in enumerate(trajectory):
            if isinstance(item, dict) and "observation" in item:
                obs_text = item["observation"].get("text", "")
                traj_lines.append(f"[Step {i//2}] OBS: {obs_text[:500]}")
            elif isinstance(item, dict) and "action_type" in item:
                traj_lines.append(f"[Step {i//2}] ACTION: {item.get('raw_prediction', '')[:200]}")
        traj_text = "\n".join(traj_lines)

        messages = [{"role": "user", "content": f"{extraction_prompt}\n\nTRAJECTORY:\n{traj_text}"}]
        try:
            response = call_llm(self.extraction_lm_config, messages)
            items = _MemoryItem.from_string(response)
            if items:
                self.memory_client.add_memories(items)
                logger.info(f"[MemoryExtraction] Saved {len(items)} memories (score={score})")
            else:
                logger.info(f"[MemoryExtraction] No memories extracted (score={score})")
        except Exception as e:
            logger.warning(f"[MemoryExtraction] Failed: {e}")  # extraction is best-effort


def construct_agent(args: argparse.Namespace) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
        # memory_client and extraction_lm_config are wired in run.py after construction
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
