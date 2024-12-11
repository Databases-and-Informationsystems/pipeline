import json
import typing

from app.pipeline.models.llm import GptModel, LLMMentionPrediction, LLMEntityPrediction
from app.model.document import DocumentEdit, Mention, Entity
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

        print(prediction_data)

        # TODO this is not good style and needs to be refactored
        for index, entity_array in enumerate(prediction_data):
            mentions = list(
                map(
                    lambda mention_index: next(
                        (m for m in document_edit.mentions if m.id == mention_index),
                        None,
                    ),
                    entity_array,
                )
            )

            if has_different_mentions(mentions):
                continue

            for mention in document_edit.mentions:
                existing_mention = next(
                    (m for m in mentions if m.id == mention.id), None
                )
                if existing_mention is not None:
                    mention.entity = Entity(id=index)

        return document_edit


def has_different_mentions(mentions: typing.List[Mention]) -> bool:
    if len(mentions) == 0:
        return False
    reference_value = mentions[0].tag

    for obj in mentions[1:]:
        if obj.tag != reference_value:
            return True
    return False
