import json

from app.pipeline.prompt_helper import PromptHelper
from app.model.document import DocumentEdit, Token, Mention
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType


class Token:
    id: int
    text: str

    def __init__(self, id: int, text: str):
        self.id = id
        self.text = text


class MentionPrediction(PipelineStep):
    def __init__(self, name: str = "MentionPrediction"):
        super().__init__(name, PipelineStepType.MENTION_PREDICTION)

    def _train(self):
        pass

    def _run(self, document_edit: DocumentEdit, schema: Schema) -> DocumentEdit:
        print("run mention detection")
        tokens = document_edit.document.tokens

        prompt = PromptHelper.get_mention_prediction_prompt(
            document_edit.document.content, document_edit.document.tokens, schema
        )
        prediction = PromptHelper.get_prediction(prompt)

        if prediction.startswith("```json") and prediction.endswith("```"):
            cleaned_prediction = prediction[7:-3].strip()
        else:
            cleaned_prediction = prediction

        try:
            prediction_data = json.loads(cleaned_prediction)
            print(prediction_data)
        except json.JSONDecodeError as e:
            print(f"JSON-Fehler: {e}")

        mentions = []
        for mention_data in prediction_data["mentions"]:
            tokens = []
            for i in range(
                mention_data["startTokenDocumentIndices"],
                mention_data["endTokenDocumentIndices"] + 1,
            ):
                tokens.append(document_edit.document.tokens[i])
            mention = Mention(tag=mention_data["type"], tokens=tokens)
            mentions.append(mention)

        document_edit.mentions = mentions

        return document_edit
