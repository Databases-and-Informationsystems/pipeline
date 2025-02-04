from flask_restx import fields
from app.controllers import api

### --------------------------------------------------------------------------------------------------------------------
# general sub -inputs
### --------------------------------------------------------------------------------------------------------------------

token_input = api.model(
    "Token",
    {
        "id": fields.Integer,
        "text": fields.String,
        "document_index": fields.Integer(required=True),
        "sentence_index": fields.Integer(required=True),
        "pos_tag": fields.String(required=True),
    },
)

mention_input = api.model(
    "Mention",
    {
        "id": fields.Integer(required=True),
        "tag": fields.String(required=True),
        "tokens": fields.List(fields.Nested(token_input, required=True)),
    },
)

entity_input = api.model(
    "Entity",
    {
        "id": fields.Integer(required=True),
        "tag": fields.String(required=True),
        "mentions": fields.List(fields.Nested(mention_input, required=True)),
    },
)

relation_input = api.model(
    "Relation",
    {
        "id": fields.Integer(required=True),
        "tag": fields.String(required=True),
        "head_mention": fields.Nested(mention_input, required=True),
        "tail_mention": fields.Nested(mention_input, required=True),
    },
)

### --------------------------------------------------------------------------------------------------------------------
# train neural network sub inputs
### --------------------------------------------------------------------------------------------------------------------

mention_document = api.model(
    "MentionDocument",
    {
        "id": fields.Integer(required=False),
        "content": fields.String(required=True),
        "tokens": fields.List(fields.Nested(token_input)),
        "mentions": fields.List(fields.Nested(mention_input)),
    },
)

document = api.model(
    "MentionDocument",
    {
        "id": fields.Integer(required=False),
        "content": fields.String(required=True),
        "tokens": fields.List(fields.Nested(token_input)),
        "mentions": fields.List(fields.Nested(mention_input)),
        "entitys": fields.List(fields.Nested(entity_input)),
        "relations": fields.List(fields.Nested(relation_input)),
    },
)

### --------------------------------------------------------------------------------------------------------------------
# Schema
### --------------------------------------------------------------------------------------------------------------------

schema_mention_input = api.model(
    "SchemaMention",
    {
        "id": fields.Integer(required=False),
        "tag": fields.String(required=True),
        "description": fields.String(required=True),
    },
)

schema_relation_input = api.model(
    "SchemaRelation",
    {
        "id": fields.Integer(required=False),
        "tag": fields.String(required=True),
        "description": fields.String(required=True),
    },
)

schema_constraint_input = api.model(
    "SchemaConstraint",
    {
        "id": fields.Integer(required=False),
        "schema_relation": fields.Nested(schema_relation_input),
        "schema_mention_head": fields.Nested(schema_mention_input),
        "schema_mention_tail": fields.Nested(schema_mention_input),
        "is_directed": fields.Boolean(required=False),
    },
)

schema_input_for_mentions = api.model(
    "Schema for Mentions",
    {
        "id": fields.Integer(required=False),
        "schema_mentions": fields.List(fields.Nested(schema_mention_input)),
    },
)

schema_input_for_relations = api.model(
    "Schema for Relations",
    {
        "id": fields.Integer(required=False),
        "schema_mentions": fields.List(fields.Nested(schema_mention_input)),
        "schema_relations": fields.List(fields.Nested(schema_relation_input)),
        "schema_constraints": fields.List(fields.Nested(schema_constraint_input)),
    },
)

### --------------------------------------------------------------------------------------------------------------------
# Tokenize
### --------------------------------------------------------------------------------------------------------------------

tokenize_step_input = api.model(
    "TokenizeInput",
    {
        "content": fields.String(required=True),
    },
)

tokenize_step_output = api.model(
    "TokenizeOutput",
    {
        "text": fields.String(required=True),
        "document_index": fields.Integer(required=True),
        "sentence_index": fields.Integer(required=True),
        "pos_tag": fields.String(required=True),
    },
)

### --------------------------------------------------------------------------------------------------------------------
# Mention detection
### --------------------------------------------------------------------------------------------------------------------

mention_step_input = api.model(
    "MentionInput",
    {
        "schema": fields.Nested(schema_input_for_mentions, required=True),
        "content": fields.String(required=True),
        "tokens": fields.List(fields.Nested(token_input)),
    },
)

new_mention = api.model(
    "NewMention",
    {
        "type": fields.String(required=True),
        "startTokenDocumentIndex": fields.Integer(required=True),
        "endTokenDocumentIndex": fields.Integer(required=True),
    },
)


### --------------------------------------------------------------------------------------------------------------------
# Entity detection
### --------------------------------------------------------------------------------------------------------------------

entity_step_input = api.model(
    "EntityInput",
    {
        "schema": fields.Nested(schema_input_for_mentions, required=True),
        "content": fields.String(required=True),
        "mentions": fields.List(fields.Nested(mention_input)),
    },
)

entity_step_output = api.model(
    "EntityOutput",
    {
        "mentions": fields.List(
            fields.Nested(
                api.model(
                    "mapped_mention",
                    {
                        "id": fields.Integer(required=True),
                        "tag": fields.String(required=True),
                        "start_token_id": fields.Integer(),
                        "end_token_id": fields.Integer(),
                        "text": fields.String(),
                    },
                ),
            ),
            required=True,  # Ensure the list itself is required
        ),
    },
)

### --------------------------------------------------------------------------------------------------------------------
# Relation detection
### --------------------------------------------------------------------------------------------------------------------

new_relation = api.model(
    "NewRelation",
    {
        "tag": fields.String(required=True),
        "head_mention_id": fields.Integer(required=True),
        "tail_mention_id": fields.Integer(required=True),
    },
)

relation_step_input = api.model(
    "RelationInput",
    {
        "schema": fields.Nested(schema_input_for_relations, required=True),
        "content": fields.String(required=True),
        "mentions": fields.List(fields.Nested(mention_input), required=True),
    },
)
### --------------------------------------------------------------------------------------------------------------------
# Train neural network
### --------------------------------------------------------------------------------------------------------------------

train_nn_input = api.model(
    "TrainNNInput",
    {
        "schema": fields.Nested(schema_input_for_relations, required=True),
        "documents": fields.List(fields.Nested(document)),
    },
)

### --------------------------------------------------------------------------------------------------------------------
# Training Results

training_results = api.model(
    "TrainingResults",
    {
        "epoch_train_loss": fields.List(fields.Float(required=True)),
        "number_of_epochs": fields.Integer(required=True),
        "cross_validation_score": fields.Float,
    },
)

### --------------------------------------------------------------------------------------------------------------------


### --------------------------------------------------------------------------------------------------------------------
# Pipeline Documentation
### --------------------------------------------------------------------------------------------------------------------

model_type_with_settings = api.model(
    "ModelTypeWithSettings",
    {
        "model_type": fields.String(required=True),
        "settings": fields.Raw(
            required=False,
            description="""
Dictionary of key value pairs, that can be added as query param to the _model_type_.
_Keys_ are always of type string.
_Values_ are of type object containing the following keys:
- **values** => possible values for the key
    - if the value is an array, all values of the array are valid (enum like behavior)
    - if the value is an string, the possible type is the value of the string
        - _"string"_ => any string can be an input
        - _"integer"_ => any integer can be an input
- **default** => the default value for the key. _Null_ if there is no default value      
                        """,
        ),
    },
)
