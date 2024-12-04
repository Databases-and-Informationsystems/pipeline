import os
import openai
from dotenv import load_dotenv
from typing import List

from app.model.document import Token
from app.model.schema import Schema


class PromptHelper:
    @staticmethod
    def get_prediction(prompt: str):
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        openai.api_key = key

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a nlp expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    @staticmethod
    def get_schema_mention_string(schema: Schema):
        schema_mention_string = ""
        for schema_mention in schema.schema_mentions:
            schema_mention_string += (
                f'"{schema_mention.tag}": {schema_mention.description}\n'
            )
        return schema_mention_string

    @staticmethod
    def get_mention_prediction_prompt(
        content: str, tokens: List[Token], schema: Schema
    ):
        schema_mention_string = PromptHelper.get_schema_mention_string(schema)

        prompt = f"""
You are an advanced text analysis assistant. Your task is to process a given text with annotated tokens and extract mentions based on their context. Each mention consists of a span of consecutive tokens that form a distinct concept or unit. Mentions should not group separate entities or actions into a single span. Each mention consists of the SMALLEST! possible, meaningful unit of connected tokens. Mentions must NEVER! combine several independent semantic units, but should be broken down as finely as possible.
Each mention should be categorized into one of the following types:

{schema_mention_string}

"mentions": "type": "<MentionType>", "startTokenDocumentIndices": <tokenDocumentIndex>, "endTokenDocumentIndices": <tokenDocumentIndex>]

Input:
- Content: The full text for reference.
- Tokens: A list of tokens with their indices, part-of-speech tags, and sentence indices.

Output:
- Mentions: A list of mentions with their MentionType und list of tokenDocumentIndices of their tokens

Extract mentions from the given text and tokens and return the output in the specified format.
Content: {content}
Tokens: {tokens}
        """

        return prompt
