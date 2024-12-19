import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns as ns
from ..model.document import Token, CMention
from ..model.schema import Schema
from ..pipeline.factory import MentionStepFactory
from ..pipeline.steps.mention_prediction import MentionStep
from ..restx_dtos import mention_step_input, new_mention


@ns.route("/mention")
class MentionStepController(Resource):

    @ns.expect(mention_step_input, validate=True)
    @ns.marshal_with(new_mention, as_list=True, code=200)
    @ns.response(400, "Invalid input")
    @ns.response(500, "Internal server error")
    def post(self):
        """
        Detect mentions in a document based on the provided input.
        """
        data = request.get_json()
        tokens_data = data.get("tokens")
        content = data.get("content")
        schema_data = data.get("schema")

        tokens: typing.List[Token] = [
            TypeAdapter(Token).validate_json(json.dumps(token_data))
            for token_data in tokens_data
        ]
        schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))

        mention_pipeline_step: MentionStep = MentionStepFactory.create(
            settings=None,
        )

        mentions: typing.List[CMention] = mention_pipeline_step.run(
            tokens=tokens, content=content, schema=schema
        )

        return [mention.model_dump(mode="json") for mention in mentions]
