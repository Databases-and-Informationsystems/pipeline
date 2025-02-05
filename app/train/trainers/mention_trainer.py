import typing
from abc import ABC, abstractmethod
from enum import Enum

from app.model.settings import ModelSize
from app.model.document import Document
from app.model.schema import Schema
from app.model.training import TrainingResults
from app.train.trainers.trainer import Trainer, TrainerStepType
from app.train.basic_nns.mention_nn import MentionBasicNN
from app.util.logger import logger


class MentionTrainModelType(Enum):
    BASIC_NEURAL_NETWORK = "basic_nn"

    @staticmethod
    def get_default() -> "MentionTrainModelType":
        return MentionTrainModelType.BASIC_NEURAL_NETWORK

    @staticmethod
    def from_string(value: str) -> "MentionTrainModelType":
        try:
            return MentionTrainModelType(value)
        except ValueError:
            return MentionTrainModelType.get_default()


class MentionTrainer(Trainer, ABC):

    def __init__(self, name: str, evaluate: bool):
        super().__init__(name, TrainerStepType.MENTION_TRAINER, evaluate)


class NNMentionTrainer(MentionTrainer):
    size: ModelSize
    nn_name: str

    def __init__(
        self,
        size: ModelSize,
        evaluate: bool,
        nn_name: str,
        name: str = "MentionTrainer",
    ):
        super().__init__(name, evaluate)
        self.size = size
        self.nn_name = nn_name

    def _train(self, schema: Schema, documents: typing.List[Document]) -> str:
        mention_nn = MentionBasicNN(
            size=self.size, documents=documents, name=self.nn_name
        )

        if mention_nn.loaded:
            raise FileNotFoundError(f"Modell bereits vorhanden: {mention_nn.name}")

        epoch_loss_list = mention_nn.start_training(documents=documents)
        mention_nn.save_as_file()

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
