import json
import typing
from abc import ABC, abstractmethod
from enum import Enum

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel, LLMRelationPrediction
from app.model.document import Mention, CEntity, CRelation
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType
from app.train.basic_nns.relation_nn import RelationBasicNN


class RelationModelType(Enum):
    LLM = "llm"
    BASIC_NEURAL_NETWORK = "basic_nn"

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
        name: str = "RelationPrediction",
    ):
        super().__init__(name)
        self.temperature = temperature
        self.gpt_model = gpt_model

    def _train(self):
        pass

    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CEntity]:

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

        return prediction_data


class NNRelationStep(RelationStep):
    model: RelationBasicNN

    def __init__(
        self,
        model: RelationBasicNN,
        name: str = "RelationPrediction",
    ):
        super().__init__(name)
        self.model = model

    def _train(self):
        pass

    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CRelation]:

        c_relations = self.model.predict(mentions=mentions)

        return c_relations
