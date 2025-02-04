import json
import typing

from flask import request
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns as ns, get_document_id, caching_enabled
from ..model.document import Mention
from ..model.schema import Schema
from ..model.settings import GptModel, Temperature
from ..pipeline.factory import RelationStepFactory, get_relation_settings
from ..pipeline.steps.relation_prediction import RelationStep, RelationModelType
from ..restx_dtos import relation_step_input, new_relation, model_type_with_settings
from ..util.file import read_json_from_file, create_file_from_data


@ns.route("/relation")
class RelationStepController(Resource):

    @ns.doc(
        description="Defines all possible _model_types_ with its possible _settings_ for the relation detection step.",
    )
    @ns.marshal_with(model_type_with_settings, as_list=True, code=200)
    def get(self):
        return [
            {
                "model_type": model_type.value,
                "settings": get_relation_settings(model_type),
            }
            for model_type in RelationModelType
        ]

    @ns.expect(relation_step_input, validate=True)
    @ns.marshal_with(new_relation, as_list=True, code=200)
    @ns.doc(
        params={
            "model_type": {
                "description": "Recommendation System that should be used.",
                "required": True,
                "enum": [mt.value for mt in RelationModelType],
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
            "name": {
                "description": "Name of the neural network. You need a trained neural network with this name",
                "required": True,
                "type": "string",
            },
        }
    )
    @ns.response(400, "Invalid input")
    @ns.response(500, "Internal server error")
    def post(self):
        """
        Detect relation in a document based on the provided input.
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

        relation_pipeline_step: RelationStep = RelationStepFactory.create(request.args)

        document_id = get_document_id(data)

        if caching_enabled():
            cached_res = read_json_from_file(
                relation_pipeline_step.pipeline_step_type, document_id
            )
            if cached_res is not None:
                return cached_res

        # TODO convert result to correct format for file / result
        res: any = relation_pipeline_step.run(
            content=content, schema=schema, mentions=mentions
        )

        if caching_enabled():
            create_file_from_data(
                res, relation_pipeline_step.pipeline_step_type, document_id
            )

        return res
