import typing

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel
from app.pipeline.steps.entity_prediction import (
    EntityPrediction,
    EntityStep,
    EntityModelType,
)
from app.pipeline.steps.relation_prediction import (
    RelationStep,
    RelationPrediction,
    RelationModelType,
)
from app.pipeline.steps.tokenizer import Tokenizer, TokenizeStep
from app.pipeline.steps.mention_prediction import (
    LLMMentionStep,
    MentionStep,
    MentionModelType,
)


class TokenizeStepFactory:

    @staticmethod
    def create() -> TokenizeStep:
        return Tokenizer()


class MentionStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> MentionStep:
        if settings.get("model_type") == "llm":
            temperature: Temperature = (
                Temperature.from_string(settings.get("temperature"))
                if settings and settings.get("temperature")
                else Temperature.get_default()
            )
            gpt_model: GptModel = (
                GptModel.from_string(settings.get("gpt-model"))
                if settings and settings.get("gpt-model")
                else GptModel.get_default()
            )
            return LLMMentionStep(
                temperature=temperature,
                gpt_model=gpt_model,
            )
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


def get_mention_settings(model_type: MentionModelType) -> dict:
    if model_type == MentionModelType.LLM:
        return {
            "temperature": {
                "values": [t.value for t in Temperature],
                "default": Temperature.get_default().value,
            },
            "gpt-model": {
                "values": [m.value for m in GptModel],
                "default": GptModel.get_default().value,
            },
        }


class EntityStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> EntityStep:
        if settings.get("model_type") == "llm":
            temperature: Temperature = (
                Temperature.from_string(settings.get("temperature"))
                if settings and settings.get("temperature")
                else Temperature.get_default()
            )
            gpt_model: GptModel = (
                GptModel.from_string(settings.get("gpt-model"))
                if settings and settings.get("gpt-model")
                else GptModel.get_default()
            )
            return EntityPrediction(
                temperature=temperature,
                gpt_model=gpt_model,
            )
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


def get_entity_settings(model_type: EntityModelType) -> dict:
    if model_type == EntityModelType.LLM:
        return {
            "temperature": {
                "values": [t.value for t in Temperature],
                "default": Temperature.get_default().value,
            },
            "gpt-model": {
                "values": [m.value for m in GptModel],
                "default": GptModel.get_default().value,
            },
        }


class RelationStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> RelationStep:
        if settings.get("model_type") == "llm":
            temperature: Temperature = (
                Temperature.from_string(settings.get("temperature"))
                if settings and settings.get("temperature")
                else Temperature.get_default()
            )
            gpt_model: GptModel = (
                GptModel.from_string(settings.get("gpt-model"))
                if settings and settings.get("gpt-model")
                else GptModel.get_default()
            )
            return RelationPrediction(
                temperature=temperature,
                gpt_model=gpt_model,
            )
        else:
            raise ValueError(
                f"model_type '{settings.get('model_type')}' is not supported."
            )


def get_relation_settings(model_type: RelationModelType) -> dict:
    if model_type == RelationModelType.LLM:
        return {
            "temperature": {
                "values": [t.value for t in Temperature],
                "default": Temperature.get_default().value,
            },
            "gpt-model": {
                "values": [m.value for m in GptModel],
                "default": GptModel.get_default().value,
            },
        }
