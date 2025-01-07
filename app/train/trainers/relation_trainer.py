import json
import typing
from abc import ABC, abstractmethod

from app.model.settings import ModelSize
from app.model.document import Document
from app.model.schema import Schema
from app.train.trainers.trainer import Trainer, TrainerStepType
from app.train.basic_nns.relation_nn import RelationBasicNN


class RelationTrainer(Trainer, ABC):

    def __init__(self, name: str, evaluate: bool):
        super().__init__(name, TrainerStepType.ENTITY_TRAINER, evaluate)

    def train(self, schema: Schema, documents: typing.List[Document]) -> str:
        res = self._train(schema, documents)
        return res

    @abstractmethod
    def _train(self, schema: Schema, documents: typing.List[Document]) -> str:
        pass


class NNRelationTrainer(RelationTrainer):
    size: ModelSize

    def __init__(
        self,
        size: ModelSize,
        evaluate: bool,
        name: str = "RelationTrainer",
    ):
        super().__init__(name, evaluate)
        self.size = size

    def _train(self, schema: Schema, documents: typing.List[Document]) -> str:
        mention_nn = RelationBasicNN(size=self.size)

        mention_nn.train(schema=schema, documents=documents)

        if self._evaluate:
            mention_nn.evaluate(schema=schema, documents=documents)

        return "hello"
