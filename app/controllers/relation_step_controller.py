import json
import typing

from flask import request, jsonify
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns as ns, get_document_id, caching_enabled
from ..model.document import Mention
from ..model.schema import Schema
from ..pipeline.factory import RelationStepFactory
from ..pipeline.steps.relation_prediction import RelationStep
from ..restx_dtos import relation_step_input, new_relation
from ..util.file import read_json_from_file, create_file_from_data


@ns.route("/relation")
class RelationStepController(Resource):

    @ns.expect(relation_step_input, validate=True)
    @ns.marshal_with(new_relation, as_list=True, code=200)
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

        relation_pipeline_step: RelationStep = RelationStepFactory.create(
            settings=None,
        )

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
