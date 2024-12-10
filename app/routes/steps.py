from flask import jsonify, request, Response
from flask_restx import Namespace, Resource, fields
from pydantic import TypeAdapter
import json

from app.model.document import DocumentEdit
from app.model.schema import Schema
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
        "id": fields.Integer(
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
        "id": fields.Integer(
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

schema_mention_model = steps_ns.model(
    "SchemaMention",
    {
        "id": fields.Integer(required=False),
        "tag": fields.String(required=True),
        "description": fields.String(required=True),
    },
)

schema_relation_model = steps_ns.model(
    "SchemaRelation",
    {
        "id": fields.Integer(required=False),
        "tag": fields.String(required=True),
        "description": fields.String(required=True),
    },
)

schema_constraint_model = steps_ns.model(
    "SchemaConstraint",
    {
        "id": fields.Integer(required=False),
        "schema_relation": fields.Nested(schema_relation_model),
        "schema_mention_head": fields.Nested(schema_mention_model),
        "schema_mention_tail": fields.Nested(schema_mention_model),
        "is_directed": fields.Boolean(required=False),
    },
)

schema_model = steps_ns.model(
    "Schema",
    {
        "id": fields.Integer(required=False),
        "schema_mentions": fields.List(fields.Nested(schema_mention_model)),
        # "_schema_relations": fields.List(fields.Nested(schema_relation_model)),
        # "_schema_constraints": fields.List(fields.Nested(schema_constraint_model)),
    },
)

combined_model = steps_ns.model(
    "CombinedModel",
    {
        "document_edit": fields.Nested(document_edit_model, required=True),
        "schema": fields.Nested(schema_model, required=True),
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


@steps_ns.route("/mention")
class MentionStepController(Resource):

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


@steps_ns.route("/entity")
class EntityStepController(Resource):

    @steps_ns.doc(
        description="Execute the entity prediction step of the pipeline.",
        responses={
            200: ("Successful response", document_edit_model),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(combined_model, validate=True)
    def post(self):
        """
        Detect entities in a document based on the provided input.
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
                step_type=PipelineStepType.ENTITY_PREDICTION,
                settings=None,
            )

            document_edit: DocumentEdit = pipeline_step.run(document_edit, schema)

            return jsonify(document_edit.model_dump(mode="json"))
        except Exception as e:
            print(e)
            return Response(str(e), status=500)
