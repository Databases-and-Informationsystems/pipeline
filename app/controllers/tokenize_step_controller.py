import typing

from flask import request
from flask_restx import Resource, ValidationError

from app.model.document import CToken
from app.pipeline.factory import TokenizeStepFactory
from app.restx_dtos import tokenize_step_input, tokenize_step_output

from . import steps_ns as ns
from ..pipeline.steps.tokenizer import TokenizeStep


@ns.route("/tokenize")
class TokenizeStepController(Resource):

    @ns.expect(tokenize_step_input, validate=True)
    @ns.marshal_with(tokenize_step_output, as_list=True, code=200)
    @ns.response(400, "Invalid input")
    @ns.response(500, "Internal server error")
    def post(self):
        """
        Tokenize a document based on the provided input.
        """
        tokenizer: TokenizeStep = TokenizeStepFactory.create()

        content = request.get_json().get("content")

        if content is None:
            raise ValidationError("Content is required")

        tokens: typing.List[CToken] = tokenizer.run(content=content)

        return [token.model_dump(mode="json") for token in tokens]
