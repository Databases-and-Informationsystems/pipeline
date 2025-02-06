import abc
import typing
from enum import Enum


class PipelineStepType(Enum):
    TOKENIZER = "tokenizer"
    MENTION_PREDICTION = "mention_prediction"
    ENTITY_PREDICTION = "entity_prediction"
    RELATION_PREDICTION = "relation_prediction"


class PipelineStep(abc.ABC):
    _pipeline_step_type: PipelineStepType
    _name: str

    def __init__(
        self,
        name: str,
        pipeline_step_type: PipelineStepType,
    ):
        self._pipeline_step_type = pipeline_step_type
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def pipeline_step_type(self) -> PipelineStepType:
        return self._pipeline_step_type

    @abc.abstractmethod
    def _get_settings(self) -> typing.Dict[str, typing.Any]:
        pass
