import typing

from app.pipeline import Pipeline
from app.model.document import DocumentEdit
from app.pipeline.models.llm import GptModel
from app.pipeline.step import PipelineStepType, PipelineStep
from app.pipeline.steps.entity_prediction import EntityPrediction
from app.pipeline.steps.tokenizer import Tokenizer
from app.pipeline.steps.mention_prediction import MentionPrediction


class PipelineFactory:

    @staticmethod
    def create(
        document_edit: DocumentEdit, settings: typing.Optional[any] = None
    ) -> Pipeline:
        pipeline_steps = [Tokenizer(), MentionPrediction()]

        return Pipeline(pipeline_steps, document_edit)

    @staticmethod
    def create_step(
        step_type: PipelineStepType, settings: typing.Optional[dict] = None
    ) -> PipelineStep:
        match step_type:
            case PipelineStepType.TOKENIZER:
                return Tokenizer()
            case PipelineStepType.MENTION_PREDICTION:
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
            case PipelineStepType.ENTITY_PREDICTION:
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
            case _:
                raise Exception(f"Unknown pipeline step type {step_type}")
