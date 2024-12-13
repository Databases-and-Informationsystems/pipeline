from flask import request, jsonify, Response
from flask_restx import Resource, Namespace
from pydantic import TypeAdapter

from app.model.document import DocumentEdit
from app.model.schema import Schema
from app.pipeline import PipelineStep
from app.pipeline.factory import PipelineFactory
from app.pipeline.step import PipelineStepType
from app.restx_dtos import document_edit_request, document_edit_model

from . import steps_ns


@steps_ns.route("/tokenize")
@steps_ns.response(400, "Invalid input")
@steps_ns.response(403, "Authorization required")
@steps_ns.response(404, "Data not found")
@steps_ns.response(500, "Internal server error")
class TokenizeStepController(Resource):

    def get(self):
        return "GET of Tokenize controller"

    @steps_ns.doc(
        description="Execute the tokenize step of the pipeline.",
        responses={
            200: ("Successful response", document_edit_model),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(document_edit_request, validate=True)
    def post(self):
        """
        Tokenize a document based on the provided input.
        """
        try:
            pipeline_step: PipelineStep = PipelineFactory.create_step(
                step_type=PipelineStepType.TOKENIZER,
                settings=None,
            )
            document_edit: DocumentEdit = pipeline_step.run(
                TypeAdapter(DocumentEdit).validate_json(request.get_data()),
                # Tokenizer does not require Schema in any case -> insert empty default Schema
                Schema(
                    schema_mentions=[],
                    schema_constraints=[],
                    schema_relations=[],
                    id=None,
                ),
            )

            return jsonify(document_edit.model_dump(mode="json"))
        except Exception as e:
            return Response(str(e), status=500)
