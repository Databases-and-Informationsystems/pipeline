import typing

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel
from app.pipeline.steps.entity_prediction import (
    EntityPrediction,
    EntityStep,
    NNEntityStep,
)
from app.pipeline.steps.relation_prediction import (
    RelationStep,
    RelationPrediction,
    NNRelationStep,
)
from app.pipeline.steps.tokenizer import Tokenizer, TokenizeStep
from app.pipeline.steps.mention_prediction import (
    LLMMentionStep,
    MentionStep,
    NNMentionStep,
)
from app.train.basic_nns.mention_nn import MentionBasicNN
from app.train.basic_nns.entity_nn import EntityBasicNN
from app.train.basic_nns.relation_nn import RelationBasicNN


class TokenizeStepFactory:

    @staticmethod
    def create() -> TokenizeStep:
        return Tokenizer()


class MentionStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> MentionStep:
        if settings.get("model_type") == "llm":
            temperature = (
                Temperature.from_string(settings.get("temperature"))
                if settings and settings.get("temperature")
                else Temperature.get_default()
            )
            model = (
                GptModel.from_string(settings.get("model"))
                if settings and settings.get("model")
                else GptModel.get_default()
            )
            return LLMMentionStep(
                temperature=temperature,
                model=model,
            )
        elif settings.get("model_type") == "basic_nn":
            model = MentionBasicNN(schema_id=settings.get("schema_id"))
            return NNMentionStep(model=model)
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


class EntityStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> EntityStep:
        if settings.get("model_type") == "llm":
            temperature = (
                Temperature.from_string(settings.get("temperature"))
                if settings and settings.get("temperature")
                else Temperature.get_default()
            )
            model = (
                GptModel.from_string(settings.get("model"))
                if settings and settings.get("model")
                else GptModel.get_default()
            )
            return EntityPrediction(
                temperature=(Temperature.from_string(temperature)),
                model=model,
            )
        elif settings.get("model_type") == "basic_nn":
            model = EntityBasicNN(schema_id=settings.get("schema_id"))
            return NNEntityStep(model=model)
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


class RelationStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> RelationStep:
        if settings.get("model_type") == "llm":
            temperature = (
                Temperature.from_string(settings.get("temperature"))
                if settings and settings.get("temperature")
                else Temperature.get_default()
            )
            model = (
                GptModel.from_string(settings.get("model"))
                if settings and settings.get("model")
                else GptModel.get_default()
            )
            return RelationPrediction(
                temperature=(Temperature.from_string(temperature)),
                model=model,
            )
        elif settings.get("model_type") == "basic_nn":
            model = RelationBasicNN(schema_id=settings.get("schema_id"))
            return NNRelationStep(model=model)
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )
