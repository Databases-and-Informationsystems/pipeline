import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns as ns, caching_enabled, get_document_id
from ..model.document import Mention, CEntity
from ..model.schema import Schema
from ..model.settings import GptModel, Temperature, PredictModelType
from ..pipeline.factory import EntityStepFactory
from ..pipeline.steps.entity_prediction import EntityStep
from ..restx_dtos import entity_step_output, entity_step_input
from ..util.file import read_json_from_file, create_file_from_data


@ns.route("/entity")
class EntityStepController(Resource):

    @ns.expect(entity_step_input, validate=True)
    # TODO fix this
    # @ns.marshal_with(entity_step_output, as_list=True, code=200)
    @ns.doc(
        params={
            "model_type": {
                "description": "Recommendation System that should be used.",
                "required": True,
                "enum": [model_type.value for model_type in PredictModelType],
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
        Detect entities in a document based on the provided input.
        """

        data = request.get_json()
        content = data.get("content")
        mentions_data = data.get("mentions")
        schema_data = data.get("schema")
        mentions: typing.List[Mention] = [
            TypeAdapter(Mention).validate_json(json.dumps(mention_data))
            for mention_data in mentions_data
        ]
        schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))

        if request.args.get("model_type") is None:
            raise ValueError("'model_type' parameter is required")

        entity_step: EntityStep = EntityStepFactory.create(
            settings={
                "model_type": request.args.get("model_type"),
                "model": request.args.get("model"),
                "temperature": request.args.get("temperature"),
            },
        )

        document_id = get_document_id(data)

        if caching_enabled():
            cached_res = read_json_from_file(
                entity_step.pipeline_step_type, document_id
            )
            if cached_res is not None:
                return cached_res

        entities: typing.List[CEntity] = entity_step.run(
            content=content, schema=schema, mentions=mentions
        )

        res = [e.to_dict() for e in entities]

        if caching_enabled():
            create_file_from_data(res, entity_step.pipeline_step_type, document_id)

        return res
