import json
import typing

from flask import request, jsonify, Response
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns
from ..model.document import Token, CMention
from ..model.schema import Schema
from ..pipeline.factory import MentionStepFactory
from ..pipeline.steps.mention_prediction import MentionStep
from ..restx_dtos import mention_step_input, mention_step_output


@steps_ns.route("/mention")
class MentionStepController(Resource):

    @steps_ns.doc(
        description="Execute the mention prediction step of the pipeline.",
        responses={
            200: ("Successful response", mention_step_output),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(mention_step_input, validate=True)
    def post(self):
        """
        Detect mentions in a document based on the provided input.
        """
        try:
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

            return jsonify([mention.model_dump(mode="json") for mention in mentions])
        except Exception as e:
            return Response(str(e), status=500)
