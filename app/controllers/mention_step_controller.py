import json

from flask import request, jsonify, Response
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns
from ..model.document import DocumentEdit
from ..model.schema import Schema
from ..pipeline import PipelineStep
from ..pipeline.factory import PipelineFactory
from ..pipeline.step import PipelineStepType
from ..restx_dtos import document_edit_model, combined_model


@steps_ns.route("/mention")
class MentionStepController(Resource):

    def get(self):
        return "GET of Mention"

    @steps_ns.doc(
        description="Execute the mention prediction step of the pipeline.",
        responses={
            200: ("Successful response", document_edit_model),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(combined_model, validate=True)
    def post(self):
        """
        Detect mentions in a document based on the provided input.
        """
        try:
            data = request.get_json()
            document_edit_data = data.get("document_edit")
            schema_data = data.get("schema")

            document_edit = TypeAdapter(DocumentEdit).validate_json(
                json.dumps(document_edit_data)
            )
            schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))

            pipeline_step: PipelineStep = PipelineFactory.create_step(
                step_type=PipelineStepType.MENTION_PREDICTION,
                settings=None,
            )

            document_edit: DocumentEdit = pipeline_step.run(document_edit, schema)

            return jsonify(document_edit.model_dump(mode="json"))
        except Exception as e:
            return Response(str(e), status=500)
