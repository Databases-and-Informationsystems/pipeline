from flask_restx import fields
from app.controllers import api

# TODO this is currently just copied from the former place.
token_model = api.model(
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

document_model = api.model(
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

document_edit_model = api.model(
    "DocumentEdit",
    {
        "document": fields.Nested(
            document_model, required=True, description="The document being edited"
        ),
    },
)

document_request = api.model(
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

document_edit_request = api.model(
    "DocumentEdit Request Body",
    {
        "document": fields.Nested(
            document_request, required=True, description="The document being edited"
        ),
    },
)

schema_mention_model = api.model(
    "SchemaMention",
    {
        "id": fields.Integer(required=False),
        "tag": fields.String(required=True),
        "description": fields.String(required=True),
    },
)

schema_relation_model = api.model(
    "SchemaRelation",
    {
        "id": fields.Integer(required=False),
        "tag": fields.String(required=True),
        "description": fields.String(required=True),
    },
)

schema_constraint_model = api.model(
    "SchemaConstraint",
    {
        "id": fields.Integer(required=False),
        "schema_relation": fields.Nested(schema_relation_model),
        "schema_mention_head": fields.Nested(schema_mention_model),
        "schema_mention_tail": fields.Nested(schema_mention_model),
        "is_directed": fields.Boolean(required=False),
    },
)

schema_model = api.model(
    "Schema",
    {
        "id": fields.Integer(required=False),
        "schema_mentions": fields.List(fields.Nested(schema_mention_model)),
    },
)

combined_model = api.model(
    "CombinedModel",
    {
        "document_edit": fields.Nested(document_edit_model, required=True),
        "schema": fields.Nested(schema_model, required=True),
    },
)
