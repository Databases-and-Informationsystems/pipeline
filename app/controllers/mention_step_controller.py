import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns as ns, caching_enabled, get_document_id
from ..model.document import Token, CMention
from ..model.schema import Schema
from ..model.settings import GptModel, Temperature
from ..pipeline.factory import MentionStepFactory
from ..pipeline.steps.mention_prediction import MentionStep
from ..restx_dtos import mention_step_input, new_mention
from ..util.file import read_json_from_file, create_file_from_data


@ns.route("/mention")
class MentionStepController(Resource):

    @ns.expect(mention_step_input, validate=True)
    @ns.marshal_with(new_mention, as_list=True, code=200)
    @ns.doc(
        params={
            "model_type": {
                "description": "Recommendation System that should be used.",
                "required": True,
                "enum": MentionStep.model_types,
            },
            "model": {
                "description": f"Open AI model (default: {GptModel.get_default().value})",
                "required": False,
                "enum": [model.value for model in GptModel],
            },
            "temperature": {
                "description": f"Temperature of the Open AI model (default: {Temperature.get_default().value})",
                "required": False,
                "enum": [temperature.value for temperature in Temperature],
            },
        }
    )
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

        if request.args.get("model_type") is None:
            raise ValueError("'model_type' parameter is required")

        tokens: typing.List[Token] = [
            TypeAdapter(Token).validate_json(json.dumps(token_data))
            for token_data in tokens_data
        ]
        schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))

        mention_pipeline_step: MentionStep = MentionStepFactory.create(
            settings={
                "model_type": request.args.get("model_type"),
                "model": request.args.get("model"),
                "temperature": request.args.get("temperature"),
            },
        )

        document_id = get_document_id(data)

        if caching_enabled():
            cached_res = read_json_from_file(
                mention_pipeline_step.pipeline_step_type, document_id
            )
            if cached_res is not None:
                return cached_res

        mentions: typing.List[CMention] = mention_pipeline_step.run(
            tokens=tokens, content=content, schema=schema
        )

        res = [mention.model_dump(mode="json") for mention in mentions]

        if caching_enabled():
            create_file_from_data(
                res, mention_pipeline_step.pipeline_step_type, document_id
            )

        return res
