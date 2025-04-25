import json
import typing
from abc import ABC, abstractmethod
from enum import Enum

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel, LLMRelationPrediction
from app.model.document import Mention, CRelation
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType
from app.util.logger import logger


class RelationModelType(Enum):
    LLM = "llm"

    @staticmethod
    def get_default():
        return RelationModelType.LLM

    @staticmethod
    def from_string(value: str) -> "RelationModelType":
        try:
            return RelationModelType(value)
        except ValueError:
            return RelationModelType.get_default()


class RelationStep(PipelineStep, ABC):

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, PipelineStepType.RELATION_PREDICTION)

    def run(self, content: str, schema: Schema, mentions: typing.List[Mention]) -> any:
        logger.info(
            f"{self.name} with settings: {self._get_settings().__str__()} executed"
        )
        res = self._run(content, schema, mentions)
        return res

    @abstractmethod
    def _run(self, content: str, schema: Schema, mentions: typing.List[Mention]) -> any:
        pass


class RelationPrediction(RelationStep):
    temperature: Temperature
    gpt_model: GptModel

    def __init__(
        self,
        temperature: Temperature,
        gpt_model: GptModel,
        name: str = "LLMRelationPrediction",
    ):
        super().__init__(name)
        self.temperature = temperature
        self.gpt_model = gpt_model

    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CRelation]:

        llm_entity_detection = LLMRelationPrediction(
            model=self.gpt_model, temperature=self.temperature
        )

        prediction_json = llm_entity_detection.run(
            content=content, schema=schema, mentions=mentions
        )

        try:
            prediction_data = json.loads(prediction_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding prediction data: {e}") from e

        logger.info(json.dumps(prediction_data, indent=2))
        try:
            c_relations: typing.List[CRelation] = [
                CRelation(**item) for item in prediction_data
            ]
        except TypeError as e:
            raise ValueError(f"Error creating CMention objects: {e}") from e

        return c_relations

    def _get_settings(self) -> typing.Dict[str, typing.Any]:
        return {
            "temperature": self.temperature,
            "gpt_model": self.gpt_model,
        }
