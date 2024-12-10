import json

from app.pipeline.models.llm import GptModel, LLMMentionPrediction, LLMEntityPrediction
from app.model.document import DocumentEdit, Mention
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType


class EntityPrediction(PipelineStep):
    temperature: float
    model: GptModel

    def __init__(
        self,
        temperature: float,
        model: GptModel,
        name: str = "MentionPrediction",
    ):
        super().__init__(name, PipelineStepType.ENTITY_PREDICTION)
        self.temperature = temperature
        self.model = model

    def _train(self):
        pass

    def _run(self, document_edit: DocumentEdit, schema: Schema) -> DocumentEdit:

        llm_entity_detection = LLMEntityPrediction(
            model=self.model, temperature=self.temperature
        )

        prediction_json = llm_entity_detection.run(
            document_edit=document_edit, schema=schema
        )

        try:
            prediction_data = json.loads(prediction_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding prediction data: {e}") from e

        # TODO add entity predictions to document edit

        return document_edit
