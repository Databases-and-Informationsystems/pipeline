import typing

from app.model.settings import Temperature
from app.pipeline.models.llm import GptModel
from app.pipeline.steps.entity_prediction import (
    EntityPrediction,
    EntityStep,
    EntityModelType,
    NNEntityStep,
)
from app.pipeline.steps.relation_prediction import (
    RelationStep,
    RelationPrediction,
    RelationModelType,
    NNRelationStep,
)
from app.pipeline.steps.tokenizer import Tokenizer, TokenizeStep
from app.pipeline.steps.mention_prediction import (
    LLMMentionStep,
    MentionStep,
    MentionModelType,
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
        model_type: MentionModelType = MentionModelType.from_string(
            settings.get("model_type")
        )
        match model_type:
            case MentionModelType.LLM:
                temperature: Temperature = Temperature.from_string(
                    settings.get("temperature")
                )
                model: GptModel = GptModel.from_string(settings.get("model"))
                return LLMMentionStep(
                    temperature=temperature,
                    model=model,
                )
            case MentionModelType.BASIC_NEURAL_NETWORK:
                mention_basic_nn: MentionBasicNN = MentionBasicNN(
                    schema_id=settings.get("schema_id")
                )
                return NNMentionStep(model=mention_basic_nn)

        raise ValueError(f"model_type '{settings.get('model_type')}' is not supported.")


def get_mention_settings(model_type: MentionModelType) -> dict:
    match model_type:
        case MentionModelType.LLM:
            return {
                "temperature": {
                    "values": [t.value for t in Temperature],
                    "default": Temperature.get_default().value,
                },
                "model": {
                    "values": [m.value for m in GptModel],
                    "default": GptModel.get_default().value,
                },
            }
        case MentionModelType.BASIC_NEURAL_NETWORK:
            # basic neural network does not have any settings yet
            return {}
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
                model: GptModel = GptModel.from_string(settings.get("model"))
                return EntityPrediction(
                    temperature=temperature,
                    model=model,
                )
            case EntityModelType.BASIC_NEURAL_NETWORK:
                entity_basic_nn: EntityBasicNN = EntityBasicNN(
                    name=settings.get("name")
                )
                return NNEntityStep(model=entity_basic_nn)
        raise ValueError(f"model_type '{settings.get('model_type')}' is not supported.")


def get_entity_settings(model_type: EntityModelType) -> dict:
    match model_type:
        case EntityModelType.LLM:
            return {
                "temperature": {
                    "values": [t.value for t in Temperature],
                    "default": Temperature.get_default().value,
                },
                "model": {
                    "values": [m.value for m in GptModel],
                    "default": GptModel.get_default().value,
                },
            }
        case EntityModelType.BASIC_NEURAL_NETWORK:
            # basic neural network does not have any settings yet
            return {}
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
                model: GptModel = GptModel.from_string(settings.get("model"))
                return RelationPrediction(
                    temperature=temperature,
                    model=model,
                )
            case RelationModelType.BASIC_NEURAL_NETWORK:
                relation_basic_nn: RelationBasicNN = RelationBasicNN(
                    schema_id=settings.get("schema_id")
                )
                return NNRelationStep(model=relation_basic_nn)
        raise ValueError(f"model_type '{settings.get('model_type')}' is not supported.")


def get_relation_settings(model_type: RelationModelType) -> dict:
    match model_type:
        case RelationModelType.LLM:
            return {
                "temperature": {
                    "values": [t.value for t in Temperature],
                    "default": Temperature.get_default().value,
                },
                "model": {
                    "values": [m.value for m in GptModel],
                    "default": GptModel.get_default().value,
                },
            }
        case RelationModelType.BASIC_NEURAL_NETWORK:
            # basic neural network does not have any settings yet
            return {}
    raise ValueError(
        f"model_type '{model_type}' is not supported for relation settings."
    )
