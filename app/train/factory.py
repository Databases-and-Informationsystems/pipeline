import typing

from app.train.trainers.mention_trainer import (
    MentionTrainer,
    NNMentionTrainer,
    MentionTrainModelType,
)
from app.train.trainers.entity_trainer import (
    EntityTrainer,
    NNEntityTrainer,
    EntityTrainModelType,
)
from app.train.trainers.relation_trainer import (
    RelationTrainer,
    NNRelationTrainer,
    RelationTrainModelType,
)
from app.model.settings import ModelSize


class MentionTrainerFactory:
    @staticmethod
    def create(settings: typing.Optional[dict]) -> MentionTrainer:
        model_type: MentionTrainModelType = MentionTrainModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case MentionTrainModelType.BASIC_NEURAL_NETWORK:
                size = ModelSize.from_string(settings.get("model_size"))
                evaluate = settings.get("enable_evaluation").lower() == "true"
                nn_name = settings.get("name")
                return NNMentionTrainer(size=size, evaluate=evaluate, nn_name=nn_name)
        raise ValueError(
            f"model_type '{settings.get('model_type')}' is not supported for mention training."
        )


def get_mention_train_settings(model_type: MentionTrainModelType) -> dict:
    match model_type:
        case MentionTrainModelType.BASIC_NEURAL_NETWORK:
            return {
                "model_size": {
                    "values": [model_size.value for model_size in ModelSize],
                    "default": ModelSize.get_default().value,
                },
                "model": {
                    "values": "boolean",
                    "default": True,
                },
            }
    raise ValueError(
        f"model_type '{model_type.value}' is not supported for mention settings."
    )


class EntityTrainerFactory:
    @staticmethod
    def create(settings: typing.Optional[dict]) -> EntityTrainer:
        model_type: EntityTrainModelType = EntityTrainModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case EntityTrainModelType.BASIC_NEURAL_NETWORK:
                size = ModelSize.from_string(settings.get("model_size"))
                evaluate = settings.get("enable_evaluation").lower() == "true"
                nn_name = settings.get("name")
                return NNEntityTrainer(size=size, evaluate=evaluate, nn_name=nn_name)

        raise ValueError(
            f"model_type '{settings.get('model_type')}' is not supported for entity training."
        )


def get_entity_train_settings(model_type: EntityTrainModelType) -> dict:
    match model_type:
        case EntityTrainModelType.BASIC_NEURAL_NETWORK:
            return {
                "model_size": {
                    "values": [model_size.value for model_size in ModelSize],
                    "default": ModelSize.get_default().value,
                },
                "model": {
                    "values": "boolean",
                    "default": True,
                },
            }
    raise ValueError(
        f"model_type '{model_type.value}' is not supported for mention settings."
    )


class RelationTrainerFactory:
    @staticmethod
    def create(settings: typing.Optional[dict]) -> RelationTrainer:
        model_type: RelationTrainModelType = RelationTrainModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case RelationTrainModelType.BASIC_NEURAL_NETWORK:
                size = ModelSize.from_string(settings.get("model_size"))
                evaluate = settings.get("enable_evaluation").lower() == "true"
                nn_name = settings.get("name")
                return NNRelationTrainer(size=size, evaluate=evaluate, nn_name=nn_name)

        raise ValueError(
            f"model_type '{settings.get('model_type')}' is not supported for relation training."
        )


def get_relation_train_settings(model_type: RelationTrainModelType) -> dict:
    match model_type:
        case RelationTrainModelType.BASIC_NEURAL_NETWORK:
            return {
                "model_size": {
                    "values": [model_size.value for model_size in ModelSize],
                    "default": ModelSize.get_default().value,
                },
                "model": {
                    "values": "boolean",
                    "default": True,
                },
            }
    raise ValueError(
        f"model_type '{model_type.value}' is not supported for mention settings."
    )
