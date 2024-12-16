import typing

from app.pipeline.models.llm import GptModel
from app.pipeline.steps.entity_prediction import EntityPrediction, EntityStep
from app.pipeline.steps.relation_prediction import RelationStep, RelationPrediction
from app.pipeline.steps.tokenizer import Tokenizer, TokenizeStep
from app.pipeline.steps.mention_prediction import MentionPrediction, MentionStep


class TokenizeStepFactory:

    @staticmethod
    def create() -> TokenizeStep:
        return Tokenizer()


class MentionStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> MentionStep:
        return MentionPrediction(
            temperature=(
                settings.get("temperature")
                if settings and settings.get("temperature")
                else 0.0
            ),
            model=GptModel.from_string(
                settings.get("model")
                if settings and settings.get("model")
                else GptModel.GPT_4O_MINI.value
            ),
        )


class EntityStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> EntityStep:
        return EntityPrediction(
            temperature=(
                settings.get("temperature")
                if settings and settings.get("temperature")
                else 0.0
            ),
            model=GptModel.from_string(
                settings.get("model")
                if settings and settings.get("model")
                else GptModel.GPT_4O_MINI.value
            ),
        )


class RelationStepFactory:

    @staticmethod
    def create(settings: typing.Optional[dict]) -> RelationStep:
        return RelationPrediction(
            temperature=(
                settings.get("temperature")
                if settings and settings.get("temperature")
                else 0.0
            ),
            model=GptModel.from_string(
                settings.get("model")
                if settings and settings.get("model")
                else GptModel.GPT_4O_MINI.value
            ),
        )
