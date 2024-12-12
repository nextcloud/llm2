# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat chain
"""
import json
import re
from random import randint
from typing import Any

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.messages.ai import AIMessage



def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</?tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append(func)
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
            if 'arguments' in func:
                func['args'] = func['arguments']
                del func['arguments']
            if not 'id' in func:
                func['id'] = str(randint(1, 10000000000))
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else:
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}

class ChatWithToolsProcessor:
    """
    A chat with tools processor
    """

    model: ChatLlamaCpp

    def __init__(self, runner: ChatLlamaCpp):
        self.model = runner

    def __call__(
            self,
            inputs: dict[str, Any],
    ) -> dict[str, str]:
        model_with_tools = self.model.bind_tools(json.loads(inputs['tools']))

        messages = []
        messages.append(SystemMessage(content=inputs['system_prompt']))
        for raw_message in inputs['history']:
            message = json.loads(raw_message)
            if message['role'] == 'assistant':
                messages.append(AIMessage(content=inputs['system_prompt']))
            elif message['role'] == 'human':
                messages.append(HumanMessage(content=inputs['system_prompt']))

        messages.append(HumanMessage(content=inputs['input']))

        try:
            tool_message = json.loads(inputs['tool_message'])
            messages.append(ToolMessage(content=tool_message['content'], name=tool_message['name'], tool_call_id='42'))
        except:
            pass

        response = model_with_tools.invoke(
            messages
        )
        print(response.content)
        if not response.tool_calls or len(response.tool_calls) == 0:
            response = AIMessage(**try_parse_tool_calls(response.content))
        return {'output': response.content, 'tool_calls': json.dumps(response.tool_calls)}