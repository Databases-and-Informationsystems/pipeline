import typing

from app.train.trainers.mention_trainer import MentionTrainer, NNMentionTrainer
from app.train.trainers.entity_trainer import EntityTrainer, NNEntityTrainer
from app.train.trainers.relation_trainer import RelationTrainer, NNRelationTrainer
from app.model.settings import ModelSize


class MentionTrainerFactory:
    @staticmethod
    def create(settings: typing.Optional[dict]) -> MentionTrainer:
        if settings.get("model_type") == "basic_nn":
            size = (
                ModelSize.from_string(settings.get("model_size"))
                if settings and settings.get("model_size")
                else ModelSize.get_default()
            )
            evaluate = settings.get("evaluate")
            return NNMentionTrainer(size=size, evaluate=evaluate)
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


class EntityTrainerFactory:
    @staticmethod
    def create(settings: typing.Optional[dict]) -> EntityTrainer:
        if settings.get("model_type") == "basic_nn":
            size = (
                ModelSize.from_string(settings.get("model_size"))
                if settings and settings.get("model_size")
                else ModelSize.get_default()
            )
            evaluate = settings.get("evaluate")
            return NNEntityTrainer(size=size, evaluate=evaluate)
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


class RelationTrainerFactory:
    @staticmethod
    def create(settings: typing.Optional[dict]) -> RelationTrainer:
        if settings.get("model_type") == "basic_nn":
            size = (
                ModelSize.from_string(settings.get("model_size"))
                if settings and settings.get("model_size")
                else ModelSize.get_default()
            )
            evaluate = settings.get("evaluate")
            return NNRelationTrainer(size=size, evaluate=evaluate)
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )
