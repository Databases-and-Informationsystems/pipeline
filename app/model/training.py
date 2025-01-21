import typing
from pydantic import BaseModel


class TrainingResults(BaseModel):
    epoch_train_loss: typing.List[float]
    number_of_epochs: int
    cross_validation_score: typing.Optional[float]
