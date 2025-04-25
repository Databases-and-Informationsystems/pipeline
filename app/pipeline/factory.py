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
        model_type: MentionModelType = MentionModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case MentionModelType.LLM:
                temperature: Temperature = Temperature.from_string(
                    settings.get("temperature")
                )
                gpt_model: GptModel = GptModel.from_string(settings.get("model"))
                return LLMMentionStep(
                    temperature=temperature,
                    gpt_model=gpt_model,
                )

        raise ValueError(f"model_type '{settings.get('model_type')}' is not supported.")


def get_mention_settings(model_type: MentionModelType) -> dict:
    match model_type:
        case MentionModelType.LLM:
            return get_default_llm_settings()
    raise ValueError(
        f"model_type '{model_type}' is not supported for mention settings."
    )


class EntityStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> EntityStep:
        model_type: EntityModelType = EntityModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case EntityModelType.LLM:
                temperature: Temperature = Temperature.from_string(
                    settings.get("temperature")
                )
                gpt_model: GptModel = GptModel.from_string(settings.get("model"))
                return EntityPrediction(
                    temperature=temperature,
                    gpt_model=gpt_model,
                )
        raise ValueError(f"model_type '{settings.get('model_type')}' is not supported.")


def get_entity_settings(model_type: EntityModelType) -> dict:
    match model_type:
        case EntityModelType.LLM:
            return get_default_llm_settings()
    raise ValueError(f"model_type '{model_type}' is not supported for entity settings.")


class RelationStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> RelationStep:
        model_type: RelationModelType = RelationModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case RelationModelType.LLM:
                temperature: Temperature = Temperature.from_string(
                    settings.get("temperature")
                )
                gpt_model: GptModel = GptModel.from_string(settings.get("model"))
                return RelationPrediction(
                    temperature=temperature,
                    gpt_model=gpt_model,
                )
        raise ValueError(f"model_type '{settings.get('model_type')}' is not supported.")


def get_relation_settings(model_type: RelationModelType) -> dict:
    match model_type:
        case RelationModelType.LLM:
            return get_default_llm_settings()
    raise ValueError(
        f"model_type '{model_type}' is not supported for relation settings."
    )


def get_default_llm_settings():
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
