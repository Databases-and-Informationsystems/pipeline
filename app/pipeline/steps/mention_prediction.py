import json
import typing
from abc import ABC, abstractmethod
from enum import Enum

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel, LLMMentionPrediction
from app.model.document import CMention, Token
from app.model.schema import Schema
from app.pipeline.step import PipelineStep, PipelineStepType
from app.train.basic_nns.mention_nn import MentionBasicNN


class MentionModelType(Enum):
    LLM = "llm"
    BASIC_NEURAL_NETWORK = "basic_nn"

    @staticmethod
    def get_default():
        return MentionModelType.LLM

    @staticmethod
    def from_string(value: str) -> "MentionModelType":
        try:
            return MentionModelType(value)
        except ValueError:
            return MentionModelType.get_default()


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


class LLMMentionStep(MentionStep):
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

        print("Temperature: " + str(temperature.value))

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


class NNMentionStep(MentionStep):
    model: MentionBasicNN

    def __init__(
        self,
        model: MentionBasicNN,
        name: str = "MentionPrediction",
    ):
        super().__init__(name)
        self.model = model

    def _train(self):
        pass

    def _run(
        self, content: str, schema: Schema, tokens: typing.List[Token]
    ) -> typing.List[CMention]:

        c_mentions = self.model.predict(tokens=tokens)

        return c_mentions
