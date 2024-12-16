import typing

from flask import request, jsonify, Response
from flask_restx import Resource

from app.model.document import CToken
from app.pipeline.factory import TokenizeStepFactory
from app.restx_dtos import tokenize_step_input, tokenize_step_output

from . import steps_ns
from ..pipeline.steps.tokenizer import TokenizeStep


@steps_ns.route("/tokenize")
@steps_ns.response(400, "Invalid input")
@steps_ns.response(403, "Authorization required")
@steps_ns.response(404, "Data not found")
@steps_ns.response(500, "Internal server error")
class TokenizeStepController(Resource):

    @steps_ns.doc(
        description="Execute the tokenize step of the pipeline.",
        responses={
            200: ("Successful response", tokenize_step_output),
            500: "Internal Server Error",
        },
    )
    @steps_ns.expect(tokenize_step_input, validate=True)
    def post(self):
        """
        Tokenize a document based on the provided input.
        """
        try:
            tokenizer: TokenizeStep = TokenizeStepFactory.create()

            content = request.get_json().get("content")

            tokens: typing.List[CToken] = tokenizer.run(content=content)

            return jsonify([token.model_dump(mode="json") for token in tokens])
        except Exception as e:
            return Response(str(e), status=500)
