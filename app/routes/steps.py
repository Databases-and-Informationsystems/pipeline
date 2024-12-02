from flask import jsonify, request, Response
from flask_restx import Namespace, Resource, fields
from pydantic import TypeAdapter

from app.model.document import DocumentEdit
from app.pipeline.factory import PipelineFactory
from app.pipeline.step import PipelineStepType, PipelineStep

steps_ns: Namespace = Namespace("steps", description="Execute single pipeline steps")


token_model = steps_ns.model(
    "Token",
    {
        "text": fields.String(
            required=True, description="Text represented by the token"
        ),
        "document_index": fields.Integer(
            required=True, description="Index in document"
        ),
        "sentence_index": fields.Integer(
            required=True, description="Sentence in document"
        ),
        "pos_tag": fields.String(
            required=True, description="Part of speech tag of the token in the document"
        ),
    },
)

document_model = steps_ns.model(
    "Document",
    {
        "id": fields.String(
            required=False, description="The unique identifier for the document"
        ),
        "content": fields.String(
            required=True, description="The content of the document"
        ),
        "tokens": fields.List(
            fields.Nested(token_model),
            required=True,
            description="List of tokens in the document. (Are not required for the tokenize pipeline step",
        ),
    },
)

document_edit_model = steps_ns.model(
    "DocumentEdit",
    {
        "document": fields.Nested(
            document_model, required=True, description="The document being edited"
        ),
    },
)

document_request = steps_ns.model(
    "Document Request Body",
    {
        "id": fields.String(
            required=False, description="The unique identifier for the document"
        ),
        "content": fields.String(
            required=True, description="The content of the document"
        ),
    },
)

document_edit_request = steps_ns.model(
    "DocumentEdit Request Body",
    {
        "document": fields.Nested(
            document_request, required=True, description="The document being edited"
        ),
    },
)


@steps_ns.route("/")
class PipelineStepController(Resource):
    def get(self):
        return jsonify({"general pipeline step": "Not implemented yet"})


@steps_ns.route("/tokenize")
class TokenizeStepController(Resource):

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
                TypeAdapter(DocumentEdit).validate_json(request.get_data())
            )

            return jsonify(document_edit.model_dump(mode="json"))
        except Exception as e:
            return Response(str(e), status=500)
