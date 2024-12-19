import abc
from enum import Enum


class PipelineStepType(Enum):
    TOKENIZER = 1
    MENTION_PREDICTION = 2
    ENTITY_PREDICTION = 3
    RELATION_PREDICTION = 4


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
