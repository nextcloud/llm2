# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processor for core:text2text:multimodal-chatwithtools."""
from __future__ import annotations

import json
import logging
import pprint
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessage
from nc_py_api import AsyncNextcloudApp

from chatwithtools import (
    build_streaming_payload,
    generate_tool_call,
    try_parse_tool_calls,
)
from streaming import StreamContext, run_runnable_with_streaming
from task_files import build_attachment_content_parts

logger = logging.getLogger(__name__)


async def resolve_message_content(nc: AsyncNextcloudApp, content: Any) -> str | list[dict[str, Any]]:
    """Resolve history content: strings pass through; file parts are fetched."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Invalid message history content")

    resolved: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError("Invalid message history content")
        if item["type"] != "file":
            resolved.append(item)
            continue
        file_id = item.get("file_id")
        if file_id is None:
            raise ValueError("Invalid message history content")
        try:
            resolved.extend(await build_attachment_content_parts(nc, file_id))
        except Exception as e:
            logger.warning("Could not build file content from id: %s. Error: %s", file_id, e)
    return resolved


async def build_input_attachment_parts(
    nc: AsyncNextcloudApp,
    attachments: list[Any],
) -> list[dict[str, Any]]:
    """Fetch current-turn input_attachments and convert to content parts."""
    parts: list[dict[str, Any]] = []
    for attachment in attachments:
        file_id = attachment
        if isinstance(attachment, dict):
            file_id = attachment.get("id", attachment.get("fileId", attachment.get("file_id")))
        parts.extend(await build_attachment_content_parts(nc, file_id))
    return parts


class MultimodalChatWithToolsProcessor:
    """Chat with tools plus image/text file attachments on the current turn and in history."""

    model: BaseChatModel

    def __init__(self, runner: BaseChatModel):
        self.model = runner

    async def _process_single_input(
            self,
            input_data: dict[str, Any],
            context: StreamContext | None = None,
    ) -> dict[str, Any]:
        if context is None or context.nc is None:
            raise ValueError("StreamContext with Nextcloud client is required for multimodal chat")

        if context.user_id:
            await context.nc.set_user(context.user_id)

        system_prompt = """
{downstream_system_prompt}

You have tools at your disposal that you can call on behalf of the user.
You can call a tool by responding with a tool call.
A tool call starts with an opening `tool_call` xml tag, then a JSON object with the name of the function and the arguments, and finally it ends with a closing `tool_call` xml tag.
It looks like this, for example:
{tool_call_example}

Here is a second example:
{tool_call_example2}

When calling tools, do not output anything else, except the tool call. Do not add sample output of the tool call. Do not output the result of the tool call yourself.
The following is a JSON specification of the tools you can call and their parameters.
{tools}
""".format(
            tools=input_data['tools'],
            downstream_system_prompt=input_data['system_prompt'],
            tool_call_example='<tool_call>{"name": "the_function_to_call", "arguments": {"param1": "the first argument", "param2": "second argument"}}</tool_call>',
            tool_call_example2='<tool_call>{"name": "search_the_web", "arguments": {"search_query": "Frank Sinatra"}}</tool_call>'
        )

        messages = [SystemMessage(content=system_prompt)]

        for raw_message in input_data['history']:
            message = json.loads(raw_message)
            content = await resolve_message_content(context.nc, message['content'])
            if message['role'] == 'assistant':
                if content == '' and message.get("tool_calls"):
                    messages.append(AIMessage(content=generate_tool_call(message['tool_calls'][0])))
                else:
                    messages.append(AIMessage(content=content))
            elif message['role'] == 'human':
                messages.append(HumanMessage(content=content))

        attachments = input_data['input_attachments']
        attachment_parts = await build_input_attachment_parts(context.nc, attachments)

        if input_data['input'] != '':
            content: list[dict[str, Any]] | str
            if attachment_parts:
                content = list(attachment_parts)
                content.append({"type": "text", "text": input_data['input']})
            else:
                content = input_data['input']
            messages.append(HumanMessage(content=content))
        elif 'tool_message' in input_data and input_data['tool_message'] != '':
            try:
                tool_messages = json.loads(input_data['tool_message'])
                for tool_message in tool_messages:
                    message_content = """
    The result of your tool call for the tool "{tool_name}" is the following:
    
    ===
    {tool_call_result}
    ===
    
    You can now formulate this in natural language for the user. Do not mention that you called a tool.
    """.format(tool_call_result=tool_message['content'], tool_name=tool_message['name'])
                    messages.append(HumanMessage(content=message_content))
            except json.JSONDecodeError as e:
                print('Failed to parse tool message')
                print(e)
        elif attachment_parts:
            messages.append(HumanMessage(content=attachment_parts))
        else:
            messages.append(HumanMessage(content=''))

        # Can't print as images will fill up too much
        # pprint.pprint(messages)
        reasoning_sink: dict[str, str] = {}
        response_content = await run_runnable_with_streaming(
            self.model,
            messages,
            context,
            stream_payload_transform=build_streaming_payload,
            suppress_empty_stream_updates=True,
            reasoning_sink=reasoning_sink,
        )

        response = AIMessage(**try_parse_tool_calls(response_content))

        return {
            'output': response.content,
            'tool_calls': json.dumps(response.tool_calls),
            'output_attachments': [],
            'reasoning': reasoning_sink.get('reasoning', ''),
        }

    async def __call__(self, inputs: dict[str, Any], context: StreamContext | None = None) -> dict[str, Any]:
        return await self._process_single_input(inputs, context)
