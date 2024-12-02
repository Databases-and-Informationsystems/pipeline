import typing

from app.pipeline import Pipeline
from app.model.document import DocumentEdit
from app.pipeline.step import PipelineStepType, PipelineStep
from app.pipeline.steps.tokenizer import Tokenizer


class PipelineFactory:

    @staticmethod
    def create(
        document_edit: DocumentEdit, settings: typing.Optional[any] = None
    ) -> Pipeline:
        pipeline_steps = [Tokenizer()]

        return Pipeline(pipeline_steps, document_edit)

    @staticmethod
    def create_step(
        step_type: PipelineStepType, settings: typing.Optional[any] = None
    ) -> PipelineStep:
        match step_type:
            case PipelineStepType.TOKENIZER:
                return Tokenizer()
            case _:
                raise Exception(f"Unknown pipeline step type {step_type}")
