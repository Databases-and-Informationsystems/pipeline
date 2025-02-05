import abc
from abc import abstractmethod
from enum import Enum

import typing

from app.model.document import Document
from app.model.schema import Schema
from app.util.logger import logger


class TrainerStepType(Enum):
    MENTION_TRAINER = "mention_trainer"
    ENTITY_TRAINER = "entity_trainer"
    RELATION_TRAINER = "relation_trainer"


class Trainer(abc.ABC):
    _trainer_step_type: TrainerStepType
    _name: str
    _evaluate: bool

    def __init__(
        self,
        name: str,
        trainer_step_type: TrainerStepType,
        evaluate: bool,
    ):
        self._trainer_step_type = trainer_step_type
        self._name = name
        self._evaluate = evaluate

    def train(self, schema: Schema, documents: typing.List[Document]) -> str:
        logger.info(
            f"Train {self.trainer_type} {{'name': '{self.name}', 'evaluate': '{self._evaluate}'}}"
        )
        res = self._train(schema, documents)
        return res

    @abstractmethod
    def _train(self, schema: Schema, documents: typing.List[Document]) -> str:
        pass

    @property
    def name(self):
        return self._name

    @property
    def trainer_type(self) -> TrainerStepType:
        return self._trainer_step_type
