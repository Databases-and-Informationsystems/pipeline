import json
import traceback
import typing

from flask import request, jsonify, Response
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns
from ..model.document import Mention
from ..model.schema import Schema
from ..pipeline.factory import RelationStepFactory
from ..pipeline.steps.relation_prediction import RelationStep
from ..restx_dtos import relation_step_input, relation_step_output


@steps_ns.route("/relation")
class RelationStepController(Resource):

    @steps_ns.doc(
        description="Execute the relation prediction step of the pipeline.",
        responses={
            200: ("Successful response", relation_step_output),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(relation_step_input, validate=True)
    def post(self):
        """
        Detect relation in a document based on the provided input.
        """
        try:
            # TODO convert to new structure
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

            # TODO
            res: any = relation_pipeline_step.run(
                content=content, schema=schema, mentions=mentions
            )

            # TODO
            return jsonify(res)
        except Exception as e:
            # TODO use a global exception wrapper to return different details in prod/dev
            error_message = str(e)
            trace = traceback.format_exc()
            return Response(f"{error_message}\n\n{trace}", status=500)
