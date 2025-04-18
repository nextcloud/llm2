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
from langchain_core.messages.ai import AIMessage

def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\w*?(.+)?\w*?</?tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            try:
                func = json.loads(m.group(1))
            except json.JSONDecodeError as e:
                func = json.loads(m.group(1)[0:-1])
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
	A chat with tools processor that supports batch processing
	"""

    model: ChatLlamaCpp

    def __init__(self, runner: ChatLlamaCpp):
        self.model = runner

    def _process_single_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        system_prompt = """
{downstream_system_prompt}

You have tools at your disposal that you can call on behalf of the user.
You can call a tool by responding with a tool call.
A tool call is a JSON object with the name of the function and the arguments, surrounded by `tool_call` xml tags.
It can look like this:
{tool_call_example}

When calling tools, do not output anything else, except the tool call. Do not add sample output of the tool call. Do not output the result of the tool call yourself.
The following is a JSON specification of the tools you can call and their parameters.
{tools}
""".format(
            tools=input_data['tools'],
            downstream_system_prompt=input_data['system_prompt'],
            tool_call_example='<tool_call>{"name": "the_function_to_call", "arguments": {"param1": "the first argument", "param2": "second argument"}}</tool_call>'
        )

        messages = []
        messages.append(SystemMessage(content=system_prompt))

        for raw_message in input_data['history']:
            message = json.loads(raw_message)
            if message['role'] == 'assistant':
                messages.append(AIMessage(content=message['content']))
            elif message['role'] == 'human':
                messages.append(HumanMessage(content=message['content']))

        messages.append(HumanMessage(content=input_data['input']))

        try:
            tool_messages = json.loads(input_data['tool_message'])
            for tool_message in tool_messages:
                message_content = """
The result of your tool call for the tool "{tool_name}" is the following:
{tool_call_result}
You can now formulate this in natural language for the user. Do not mention that you called a tool.
""".format(tool_call_result=tool_message['content'], tool_name=tool_message['name'])
                messages.append(HumanMessage(
                    content=message_content
                ))
        except json.JSONDecodeError as e:
            print('Failed to parse tool message')
            print(e)

        response = self.model.invoke(messages)

        #if not response.tool_calls or len(response.tool_calls) == 0:
        response = AIMessage(**try_parse_tool_calls(response.content))

        return {
            'output': response.content,
            'tool_calls': json.dumps(response.tool_calls)
        }

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._process_single_input(inputs)