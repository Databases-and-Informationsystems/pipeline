import json
import typing
from abc import ABC, abstractmethod

from app.model.settings import ModelSize
from app.model.document import Document
from app.model.schema import Schema
from app.model.training import TrainingResults
from app.train.trainers.trainer import Trainer, TrainerStepType
from app.train.basic_nns.mention_nn import MentionBasicNN


class MentionTrainer(Trainer, ABC):

    def __init__(self, name: str, evaluate: bool):
        super().__init__(name, TrainerStepType.MENTION_TRAINER, evaluate)

    def train(self, schema: Schema, documents: typing.List[Document]) -> str:
        res = self._train(schema, documents)
        return res

    @abstractmethod
    def _train(self, schema: Schema, documents: typing.List[Document]) -> str:
        pass


class NNMentionTrainer(MentionTrainer):
    size: ModelSize

    def __init__(
        self,
        size: ModelSize,
        evaluate: bool,
        name: str = "MentionTrainer",
    ):
        super().__init__(name, evaluate)
        self.size = size

    def _train(self, schema: Schema, documents: typing.List[Document]) -> str:
        mention_nn = MentionBasicNN(size=self.size, documents=documents)

        epoch_loss_list = mention_nn.start_training(documents=documents)
        mention_nn.save_as_file(schema_id=schema.id)

        training_results = TrainingResults(
            epoch_train_loss=epoch_loss_list,
            number_of_epochs=len(epoch_loss_list),
            cross_validation_score=None,
        )
        if self._evaluate:
            training_results.cross_validation_score = mention_nn.evaluate(
                schema=schema, documents=documents
            )

        return training_results
