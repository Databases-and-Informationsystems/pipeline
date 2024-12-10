import abc
import typing
from abc import abstractmethod
from enum import Enum
from typing import final

from app.model.document import DocumentEdit
from app.model.schema import Schema


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

    @final
    def run(self, document_edit: DocumentEdit, schema: Schema) -> DocumentEdit:
        res = self._run(document_edit, schema)
        return res

    @final
    def train(self):
        self._train()

    @abstractmethod
    def _run(self, document_edit: DocumentEdit, schema: Schema) -> DocumentEdit:
        pass

    @abstractmethod
    def _train(self):
        pass
