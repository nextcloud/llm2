# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any
from langchain.prompts import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

from lib.streaming import StreamContext


class ReformatParagraphsProcessor:
    """
    Segments text by subject changes; model returns anchor phrases only.
    """
    system_prompt = (
        "You output anchors from a continuous text based on subject changes."
    )

    user_prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""You will receive a continuous block of text without line breaks. Your task is to identify points in the text where the subject or topic changes (e.g., a shift to a new person, place, concept, or thematic focus).

Do NOT break lines based on sentence length or grammar unless the subject actually changes.

Once you have identified these segments, do NOT output the full text. Instead, for each paragraph created, output ONLY the first 3-5 words verbatim of that paragraph. These serve as anchors for programmatic retrieval.

Format your output as a plain list of these anchor words, one per line. Do not include numbers, bullet points, or any additional commentary.

Example input: "The market for electric vehicles is expanding rapidly. In contrast, traditional motorcycle sales are declining globally. Aside from transportation, the price of copper remains volatile."

Example output:
The market for electric vehicles
In contrast, traditional motorcycle
Aside from transportation, the price

---

Continuous text to segment:

{text}
""")

    runnable: Runnable

    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    @staticmethod
    def _parse_anchors_from_model_output(raw: str) -> list[str]:
        if raw == "":
            return []
        anchors: list[str] = []
        for line in raw.split("\n"):
            line = line.strip()
            if line == "":
                continue
            anchors.append(line)
        return anchors

    @staticmethod
    def _insert_paragraph_breaks_by_anchors(text: str, anchors: list[str]) -> str:
        if len(anchors) < 2:
            return text
        result = text
        search_offset = 0
        delta = 0
        for i in range(1, len(anchors)):
            anchor = anchors[i]
            pos = text.find(anchor, search_offset)
            if pos == -1:
                continue
            insert_at = pos + delta
            replace_from = insert_at
            while replace_from > 0 and result[replace_from - 1].isspace():
                replace_from -= 1
            result = result[:replace_from] + "\n\n" + result[insert_at:]
            delta += 2 - (insert_at - replace_from)
            search_offset = pos + len(anchor)
        return result

    def __call__(self, inputs: dict[str, Any], context: StreamContext) -> dict[str, Any]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt.format(
                text=inputs['input']
            ))
        ]
        output = self.runnable.invoke(messages)
        raw = output.content
        anchors = self._parse_anchors_from_model_output(raw)
        reformatted = self._insert_paragraph_breaks_by_anchors(inputs["input"], anchors)
        return {"output": reformatted}