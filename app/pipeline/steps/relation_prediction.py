import json
import typing
from abc import ABC, abstractmethod

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel, LLMRelationPrediction
from app.model.document import Mention, CEntity
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType


class RelationStep(PipelineStep, ABC):
    model_types = ["llm"]

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, PipelineStepType.RELATION_PREDICTION)

    def run(self, content: str, schema: Schema, mentions: typing.List[Mention]) -> any:
        res = self._run(content, schema, mentions)
        return res

    @abstractmethod
    def _run(self, content: str, schema: Schema, mentions: typing.List[Mention]) -> any:
        pass


class RelationPrediction(RelationStep):
    temperature: Temperature
    model: GptModel

    def __init__(
        self,
        temperature: Temperature,
        model: GptModel,
        name: str = "RelationPrediction",
    ):
        super().__init__(name)
        self.temperature = temperature
        self.model = model

    def _train(self):
        pass

    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CEntity]:

        llm_entity_detection = LLMRelationPrediction(
            model=self.model, temperature=self.temperature
        )

        prediction_json = llm_entity_detection.run(
            content=content, schema=schema, mentions=mentions
        )

        try:
            prediction_data = json.loads(prediction_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding prediction data: {e}") from e

        return prediction_data
