import abc
from enum import Enum


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

    @property
    def name(self):
        return self._name

    @property
    def trainer_type(self) -> TrainerStepType:
        return self._trainer_step_type
