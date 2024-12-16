import json
import typing
from abc import ABC, abstractmethod

from app.pipeline.models.llm import GptModel, LLMEntityPrediction
from app.model.document import Mention
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType


class EntityStep(PipelineStep, ABC):

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, PipelineStepType.ENTITY_PREDICTION)

    def run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[typing.List[int]]:
        res = self._run(content, schema, mentions)
        return res

    @abstractmethod
    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[typing.List[int]]:
        pass


class EntityPrediction(EntityStep):
    temperature: float
    model: GptModel

    def __init__(
        self,
        temperature: float,
        model: GptModel,
        name: str = "MentionPrediction",
    ):
        super().__init__(name)
        self.temperature = temperature
        self.model = model

    def _train(self):
        pass

    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[typing.List[int]]:

        llm_entity_detection = LLMEntityPrediction(
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
