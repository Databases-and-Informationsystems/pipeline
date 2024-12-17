import json
import typing
from abc import ABC, abstractmethod

from app.pipeline.models.llm import GptModel, LLMEntityPrediction
from app.model.document import Mention, CEntity
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
    ) -> typing.List[CEntity]:
        res = self._run(content, schema, mentions)
        return res

    @abstractmethod
    def _run(
        self, content: str, schema: Schema, mentions: typing.List[Mention]
    ) -> typing.List[CEntity]:
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

        # llm_entity_detection = LLMEntityPrediction(
        #    model=self.model, temperature=self.temperature
        # )

        # prediction_json = llm_entity_detection.run(
        #    content=content, schema=schema, mentions=mentions
        # )

        # try:
        #    prediction_data = json.loads(prediction_json)
        # except json.JSONDecodeError as e:
        #    raise ValueError(f"Error decoding prediction data: {e}") from e

        prediction_data = [
            [1, 2],
            [3],
            [4],
            [5],
            [6],
            [8],
            [9, 10],
            [11, 12, 13],
            [14],
            [15],
            [16],
            [17],
            [18],
            [19],
            [20],
            [21],
            [22],
            [23],
            [24],
            [25],
            [26],
            [27],
            [28],
            [29],
            [30],
            [31],
            [32],
            [33, 34],
            [35],
            [36],
        ]
        print(prediction_data)
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
