import os
import typing
from abc import ABC

from app.model.document import DocumentEdit, Token, Mention
from app.model.schema import Schema
from app.model.settings import GptModel
from app.util.llm_util import get_prediction, extract_json


class LLM(ABC):
    temperature = 0.0
    model: GptModel = GptModel.GPT_4O_MINI
    open_ai_key: str

    def __init__(self, model: GptModel = GptModel.GPT_4O_MINI, temperature=0.0):
        self.model = model
        self.temperature = temperature

        self.open_ai_key = os.getenv("OPENAI_API_KEY")
        if self.open_ai_key is None:
            raise EnvironmentError("OpenAI API key not set")

    @staticmethod
    def get_schema_mention_string(schema: Schema, for_entities: bool = False) -> str:
        schema_mentions = schema.schema_mentions
        if for_entities:
            schema_mentions = filter(
                lambda mention: mention.has_entities, schema_mentions
            )

        schema_mention_string = ""
        for schema_mention in schema_mentions:
            schema_mention_string += (
                f'"{schema_mention.tag}": {schema_mention.description}\n'
            )
        return schema_mention_string


class LLMMentionPrediction(LLM):
    def __init__(self, model: GptModel, temperature: float):
        super().__init__(model, temperature)

    def run(self, document_edit: DocumentEdit, schema: Schema) -> str:
        prompt: str = self._get_prompt(document_edit, schema)
        print(prompt)
        res = get_prediction(
            prompt=prompt,
            model=self.model,
            key=self.open_ai_key,
            temperature=self.temperature,
        )

        return extract_json(res)

    def _get_prompt(self, document_edit: DocumentEdit, schema: Schema) -> str:
        schema_mention_string = LLMMentionPrediction.get_schema_mention_string(schema)
        return f"""
You are an advanced text analysis assistant. Your task is to process a given text with annotated tokens and extract mentions based on their context. 
Each mention consists of a span of consecutive tokens that form a distinct concept or unit. 
Mentions should not group separate entities or actions into a single span. 
Each mention consists of the SMALLEST! possible, meaningful unit of connected tokens. 
Mentions must NEVER! combine several independent semantic units, but should be broken down as finely as possible.

Each mention should be categorized into one of the following types:

{schema_mention_string}

Input:
- Content: The full text for reference.
- Tokens: A list of tokens with their meanings (=text) and sentence indices.

Output:
- Mentions: A list of mentions with their MentionType and list of tokenDocumentIndices of their tokens (e.g. [{{"type": "<mention-type-1>", "startTokenIndex": 0, "endTokenIndex": 1}}] as raw json file

Example (The mention types in the example might not be available in the actual task): 
Content: "After a claim is registered, it is examined by a claims officer."
Tokens: {{
            "text": "After",
            "indexInDocument": 0,
            "posTag": "IN",
            "sentenceIndex": 0
        }},
        {{
            "id": 2,
            "text": "a",
            "indexInDocument": 1,
            "posTag": "DT",
            "sentenceIndex": 0
        }},
        {{
            "id": 3,
            "text": "claim",
            "indexInDocument": 2,
            "posTag": "NN",
            "sentenceIndex": 0
        }},
        {{
            "id": 4,
            "text": "is",
            "indexInDocument": 3,
            "posTag": "VBZ",
            "sentenceIndex": 0
        }},
        {{
            "id": 5,
            "text": "registered",
            "indexInDocument": 4,
            "posTag": "VBN",
            "sentenceIndex": 0
        }},
        {{
            "id": 6,
            "text": ",",
            "indexInDocument": 5,
            "posTag": ",",
            "sentenceIndex": 0
        }},
        {{
            "id": 7,
            "text": "it",
            "indexInDocument": 6,
            "posTag": "PRP",
            "sentenceIndex": 0
        }},
        {{
            "id": 8,
            "text": "is",
            "indexInDocument": 7,
            "posTag": "VBZ",
            "sentenceIndex": 0
        }},
        {{
            "id": 9,
            "text": "examined",
            "indexInDocument": 8,
            "posTag": "VBN",
            "sentenceIndex": 0
        }},
        {{
            "id": 10,
            "text": "by",
            "indexInDocument": 9,
            "posTag": "IN",
            "sentenceIndex": 0
        }},
        {{
            "id": 11,
            "text": "a",
            "indexInDocument": 10,
            "posTag": "DT",
            "sentenceIndex": 0
        }},
        {{
            "id": 12,
            "text": "claims",
            "indexInDocument": 11,
            "posTag": "NNS",
            "sentenceIndex": 0
        }},
        {{
            "id": 13,
            "text": "officer",
            "indexInDocument": 12,
            "posTag": "NN",
            "sentenceIndex": 0
        }},
        {{
            "id": 14,
            "text": ".",
            "indexInDocument": 13,
            "posTag": ".",
            "sentenceIndex": 0
        }},

Result:
{{
    "mentions": [
        {{
            "type": "Activity Data",
            "startTokenDocumentIndex": 1,
            "endTokenDocumentIndex": 2
        }},
        {{
            "type": "Activity",
            "startTokenDocumentIndex": 4,
            "endTokenDocumentIndex": 4
        }},
        {{
            "type": "Activity Data",
            "startTokenDocumentIndex": 6,
            "endTokenDocumentIndex": 6
        }},
        {{
            "type": "Activity",
            "startTokenDocumentIndex": 8,
            "endTokenDocumentIndex": 8
        }},
        {{
            "type": "Actor",
            "startTokenDocumentIndex": 10,
            "endTokenDocumentIndex": 12
        }}
    ]
}}

Extract mentions from the given text and tokens and return the output in the specified format.
Content: {document_edit.document.content}
Tokens: {list(map(lambda t: t.to_json(), document_edit.document.tokens))}
                """


class LLMEntityPrediction(LLM):
    def __init__(self, model: GptModel, temperature: float):
        super().__init__(model, temperature)

    def run(self, document_edit: DocumentEdit, schema: Schema) -> str:
        prompt: str = self._get_prompt(document_edit, schema)
        print(prompt)
        res = get_prediction(
            prompt=prompt,
            model=self.model,
            key=self.open_ai_key,
            temperature=self.temperature,
        )

        return extract_json(res)

    def _get_prompt(self, document_edit: DocumentEdit, schema: Schema) -> str:
        return f"""
**Context**:
You are an advanced text analysis assistant designed to process and analyze business-related texts. Your task is to identify entities in a given text by grouping mentions that refer to the same real-world entity. Each entity is a group of mentions that refer to the same thing in the text. The output should be in JSON format.

**Input Specification**:
- **Text**: A passage of text (string).
- **Mentions**: A JSON array of objects. Each object represents a mention in the text and contains:
    - id: Unique identifier for the mention.
    - tag: Type of the mention.
    - start_token_id: The id of the first token of the mention.
    - end_token_id: The id after the last token of the mention.
    - text: The exact text of the mention.
- **Mention Types**: A JSON array of the different mention types with a description of each type:
    - tag: Name of the mention type.
    - description: The description of the mention type.
    
**Output Specification**:
You must return a JSON array containing groups of mention IDs. Each group represents an entity in the text. Mentions within the same group must refer to the same real-world entity.
The output should be in a raw JSON format. 
    - Each element is a json list of ids (numbers). Each of the ids in one list refer to the same real-world entity. There might be mentions, that are the single element of a list. Is is not possible, that mentions with different tag are in the same list


**Input**
- Text:     
    {document_edit.document.content}
- Mentions:
    {LLMEntityPrediction._get_mention_list_string(document_edit.mentions, document_edit.document.tokens)}
- Mention Types:
    {LLMEntityPrediction.get_schema_mention_string(schema)}

        """

    @staticmethod
    def _get_mention_list_string(
        mentions: typing.List[Mention], tokens: typing.List[Token]
    ) -> str:
        return f"{[m.to_json(tokens) for m in mentions]}"
