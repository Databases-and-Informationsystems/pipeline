import json
import traceback
import typing

from flask import request, jsonify, Response
from flask_restx import Resource
from pydantic import TypeAdapter

from . import steps_ns
from ..model.document import Mention
from ..model.schema import Schema
from ..pipeline.factory import EntityStepFactory
from ..pipeline.steps.entity_prediction import EntityStep
from ..restx_dtos import entity_step_output, entity_step_input


@steps_ns.route("/entity")
class EntityStepController(Resource):

    @steps_ns.doc(
        description="Execute the entity prediction step of the pipeline.",
        responses={
            200: ("Successful response", entity_step_output),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(entity_step_input, validate=True)
    def post(self):
        """
        Detect entities in a document based on the provided input.
        """
        try:
            data = request.get_json()
            content = data.get("content")
            mentions_data = data.get("mentions")
            schema_data = data.get("schema")

            mentions: typing.List[Mention] = [
                TypeAdapter(Mention).validate_json(json.dumps(mention_data))
                for mention_data in mentions_data
            ]
            schema = TypeAdapter(Schema).validate_json(json.dumps(schema_data))

            entity_step: EntityStep = EntityStepFactory.create(
                settings=None,
            )

            entities: typing.List[typing.List[int]] = entity_step.run(
                content=content, schema=schema, mentions=mentions
            )

            return jsonify(entities)
        except Exception as e:
            # TODO use a global exception wrapper to return different details in prod/dev
            error_message = str(e)
            trace = traceback.format_exc()
            return Response(f"{error_message}\n\n{trace}", status=500)
