import json
import typing
from abc import ABC, abstractmethod

from app.pipeline.models.llm import GptModel, LLMMentionPrediction
from app.model.document import CMention, Token
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType


class MentionStep(PipelineStep, ABC):

    def __init__(
        self,
        name: str,
    ):
        super().__init__(name, PipelineStepType.MENTION_PREDICTION)

    def run(
        self, content: str, schema: Schema, tokens: typing.List[Token]
    ) -> typing.List[CMention]:
        res = self._run(content, schema, tokens)
        return res

    @abstractmethod
    def _run(
        self, content: str, schema: Schema, tokens: typing.List[Token]
    ) -> typing.List[CMention]:
        pass


class MentionPrediction(MentionStep):
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
        self, content: str, schema: Schema, tokens: typing.List[Token]
    ) -> typing.List[CMention]:

        llm_mention_detection = LLMMentionPrediction(
            model=self.model, temperature=self.temperature
        )

        prediction_json = llm_mention_detection.run(
            content=content, schema=schema, tokens=tokens
        )

        try:
            prediction_data = json.loads(prediction_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding prediction data: {e}") from e

        try:
            c_mentions: typing.List[CMention] = [
                CMention(**item) for item in prediction_data
            ]
        except TypeError as e:
            raise ValueError(f"Error creating CMention objects: {e}") from e

        return c_mentions
