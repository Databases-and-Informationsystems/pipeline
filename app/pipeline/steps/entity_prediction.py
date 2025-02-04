import json
import typing
from abc import ABC, abstractmethod
from enum import Enum

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel, LLMEntityPrediction
from app.model.document import Mention, CEntity
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType
from app.train.basic_nns.entity_nn import EntityBasicNN


class EntityModelType(Enum):
    LLM = "llm"
    BASIC_NEURAL_NETWORK = "basic_nn"

    @staticmethod
    def get_default():
        return EntityModelType.LLM

    @staticmethod
    def from_string(value: str) -> "EntityModelType":
        try:
            return EntityModelType(value)
        except ValueError:
            return EntityModelType.get_default()


class EntityStep(PipelineStep, ABC):

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, PipelineStepType.ENTITY_PREDICTION)

    def run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CEntity]:
        res = self._run(content, schema, mentions)
        return res

    @abstractmethod
    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CEntity]:
        pass


class EntityPrediction(EntityStep):
    temperature: Temperature
    model: GptModel

    def __init__(
        self,
        temperature: Temperature,
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

        entities: typing.List[CEntity] = [
            get_mentions(indices, mentions) for indices in prediction_data
        ]

        return entities


def get_mentions(indices: typing.List[int], mentions: typing.List[Mention]) -> CEntity:
    mention_dict = {mention.id: mention for mention in mentions}

    # Collect mentions matching the given indices
    res_mentions = []
    for index in indices:
        if index not in mention_dict:
            raise ValueError(f"No mention found with id: {index}")
        res_mentions.append(mention_dict[index])

    return CEntity(mentions=res_mentions)


class NNEntityStep(EntityStep):
    model: EntityBasicNN

    def __init__(
        self,
        model: EntityBasicNN,
        name: str = "MentionPrediction",
    ):
        super().__init__(name)
        self.model = model

    def _train(self):
        pass

    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[typing.List[int]]:
        c_entities = self.model.predict(mentions=mentions)

        return c_entities
