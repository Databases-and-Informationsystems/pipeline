import json

from app.pipeline.models.llm import GptModel, LLMMentionPrediction
from app.model.document import DocumentEdit, Mention
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType


class MentionPrediction(PipelineStep):
    temperature: float
    model: GptModel

    def __init__(
        self,
        temperature: float,
        model: GptModel,
        name: str = "MentionPrediction",
    ):
        super().__init__(name, PipelineStepType.MENTION_PREDICTION)
        self.temperature = temperature
        self.model = model

    def _train(self):
        pass

    def _run(self, document_edit: DocumentEdit, schema: Schema) -> DocumentEdit:

        llm_mention_detection = LLMMentionPrediction(
            model=self.model, temperature=self.temperature
        )

        prediction_json = llm_mention_detection.run(
            document_edit=document_edit, schema=schema
        )

        try:
            prediction_data = json.loads(prediction_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding prediction data: {e}") from e

        mentions = []
        for mention_data in prediction_data["mentions"]:
            tokens = []
            for i in range(
                mention_data["startTokenDocumentIndex"],
                mention_data["endTokenDocumentIndex"] + 1,
            ):
                tokens.append(document_edit.document.tokens[i])
            mention = Mention(tag=mention_data["type"], tokens=tokens)
            mentions.append(mention)

        document_edit.mentions = mentions

        return document_edit
