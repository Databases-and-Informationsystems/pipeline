import typing

from app.model.document import DocumentEdit
from app.pipeline.step import PipelineStep


class Pipeline:
    _pipeline_steps: typing.List[PipelineStep]
    _document_edit: DocumentEdit

    def __init__(
        self, pipeline_steps: typing.List[PipelineStep], document_edit: DocumentEdit
    ):
        if not pipeline_steps:
            raise ValueError("Pipeline must have at least one step.")
        self._document_edit = document_edit
        self._pipeline_steps = pipeline_steps
        self._current_index = 0

    def get_pipeline_steps(self) -> typing.List[PipelineStep]:
        return self._pipeline_steps
