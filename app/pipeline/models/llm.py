import os
import typing
from abc import ABC

from app.model.document import Token, Mention
from app.model.schema import Schema
from app.model.settings import GptModel, Temperature
from app.util.llm_util import get_prediction, extract_json
from app.util.logger import logger


class LLM(ABC):
    temperature = 0.0
    model: GptModel = GptModel.GPT_4O_MINI
    open_ai_key: str

    def __init__(
        self, model: GptModel = GptModel.GPT_4O_MINI, temperature=Temperature.NONE
    ):
        self.model = model
        self.temperature = temperature.to_float()

        self.open_ai_key = os.getenv("OPENAI_API_KEY")
        if self.open_ai_key is None:
            raise EnvironmentError("OpenAI API key not set")

    @staticmethod
    def get_schema_mention_string(schema: Schema, for_entities: bool = False) -> str:
        return "\n".join(
            f'"{schema_mention.tag}": {schema_mention.description}'
            for schema_mention in filter(
                lambda mention: mention.has_entities if for_entities else True,
                schema.schema_mentions,
            )
        )

    @staticmethod
    def get_schema_relation_string(schema: Schema) -> str:
        return "\n".join(
            f'"{schema_relation.tag}": {schema_relation.description}'
            for schema_relation in schema.schema_relations
        )

    @staticmethod
    def get_schema_constraint_string(schema: Schema) -> str:
        return "\n".join(
            f"""
{{
    relation: {schema_constraint.schema_relation.tag},
    head_mention: {schema_constraint.schema_mention_head.tag}
    tail_mention: {schema_constraint.schema_mention_tail.tag}
    is_directed: {schema_constraint.is_directed}
}}
"""
            for schema_constraint in schema.schema_constraints
        )


class LLMMentionPrediction(LLM):
    def __init__(self, model: GptModel, temperature: Temperature):
        super().__init__(model, temperature)

    def run(self, content: str, schema: Schema, tokens: typing.List[Token]) -> str:
        prompt: str = LLMMentionPrediction._get_prompt(content, schema, tokens)
        logger.debug(f"Open AI mention prediction with input:\n{prompt}")
        res = get_prediction(
            prompt=prompt,
            model=self.model,
            key=self.open_ai_key,
            temperature=self.temperature,
        )
        logger.debug(f"Open AI mention prediction output:\n{res}")
        return extract_json(res)

    @staticmethod
    def _get_prompt(content: str, schema: Schema, tokens: typing.List[Token]) -> str:
        schema_mention_string = LLMMentionPrediction.get_schema_mention_string(schema)
        return f"""
You are an advanced text analysis assistant. Your task is to process a given text with annotated tokens and extract mentions based on their context. 
Each mention consists of a span of consecutive tokens that form a distinct concept or unit. 
Mentions should not group separate entities or actions into a single span. 
Each mention consists of the SMALLEST! possible, meaningful unit of connected tokens. Nevertheless, articles are assigned to their corresponding nouns.
 
Mentions must NEVER! combine several independent semantic units, but should be broken down as finely as possible.

Each mention should be categorized into one of the following types:

{schema_mention_string}

Input:
- Content: The full text for reference.
- Tokens: A list of tokens with their meanings (=text) and sentence indices.

Output:
- Mentions: A list of mentions with their MentionType and list of tokenDocumentIndices of their tokens (e.g. [{{"type": "<mention-type-1>", "startTokenIndex": 0, "endTokenIndex": 1}}] as raw json file. The mentions must not overlap under any circumstances.

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
[
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

Extract mentions from the given text and tokens and return the output in the specified format.
Content: {content}
Tokens: {list(map(lambda t: t.to_json(), tokens))}
                """


class LLMEntityPrediction(LLM):
    def __init__(self, model: GptModel, temperature: Temperature):
        super().__init__(model, temperature)

    def run(self, content: str, schema: Schema, mentions: typing.List[Mention]) -> str:
        prompt: str = LLMEntityPrediction._get_prompt(content, schema, mentions)
        logger.debug(f"Open AI entity prediction with input:\n'{prompt}'")
        res = get_prediction(
            prompt=prompt,
            model=self.model,
            key=self.open_ai_key,
            temperature=self.temperature,
        )
        logger.debug(f"Open AI entity prediction output:\n{res}")

        return extract_json(res)

    @staticmethod
    def _get_prompt(
        content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> str:
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
    {content}
- Mentions:
    {LLMEntityPrediction._get_mention_list_string(mentions)}
- Mention Types:
    {LLMEntityPrediction.get_schema_mention_string(schema)}

        """

    @staticmethod
    def _get_mention_list_string(mentions: typing.List[Mention]) -> str:
        return f"{[m.to_json() for m in mentions]}"


class LLMRelationPrediction(LLM):
    def __init__(self, model: GptModel, temperature: Temperature):
        super().__init__(model, temperature)

    def run(self, content: str, schema: Schema, mentions: typing.List[Mention]) -> str:
        prompt: str = self._get_prompt(content, schema, mentions)
        logger.debug(f"Open AI relation prediction with input:\n{prompt}")
        res = get_prediction(
            prompt=prompt,
            model=self.model,
            key=self.open_ai_key,
            temperature=self.temperature,
        )
        logger.debug(f"Open AI relation prediction output:\n{res}")

        return extract_json(res)

    def _get_prompt(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> str:
        return f"""
**Context**:
You are an advanced text analysis assistant designed to process and analyze business-related texts. Your task is to identify relations between mentions with a given text and given mentions. A relation describes a specific relationship between 2 mentions. The output should be in JSON format.

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
- **Relation Types**: A JSON array of the different relation types with a description of each type:
    - tag: Name of the relation type.
    - description: The description of the relation type.
- **Constraints**: A JSON array of constraints. Each Relation in the result must match a constraint from this array:
    - relation: The tag of the relation.
    - head_mention: The type of the mention that is allowed at the head of the relation.
    - tail_mention: The type of the mention that is allowed at the tail of the relation.
    - is_directed: "true", if head_mention and tail_mention can be changed
    
**Output Specification**:
    - You must return a JSON array containing relations. Each relation contains of a 'tag' (relates to the tag of a relation type), 'head_mention_id' (relates to the id of a mention from the input list), 'tail_mention_id' (relates to the id of a mention from the input list).
        - Each entry in the array must fulfill at least one constraint.
        
**Input**
- Text:
    {content}
- Mentions:
    {LLMRelationPrediction._get_mention_list_string(mentions)}
- Mention Types:
    {LLMRelationPrediction.get_schema_mention_string(schema)}
- Relation Types:
    {LLMRelationPrediction.get_schema_relation_string(schema)}
- Constraints
    {LLMRelationPrediction.get_schema_constraint_string(schema)}
        """

    @staticmethod
    def _get_mention_list_string(mentions: typing.List[Mention]) -> str:
        return f"{[m.to_json() for m in mentions]}"
