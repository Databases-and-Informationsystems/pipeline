import typing

from flask import request
from flask_restx import Resource, ValidationError

from app.model.document import CToken
from app.pipeline.factory import TokenizeStepFactory
from app.restx_dtos import tokenize_step_input, tokenize_step_output

from . import steps_ns as ns, get_document_id, caching_enabled
from ..pipeline.steps.tokenizer import TokenizeStep
from ..util.file import read_json_from_file, create_caching_file_from_data


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

        data = request.get_json()
        content = data.get("content")

        if content is None:
            raise ValidationError("Content is required")

        document_id = get_document_id(data)

        if caching_enabled():
            cached_res = read_json_from_file(tokenizer.pipeline_step_type, document_id)
            if cached_res is not None:
                return cached_res

        tokens: typing.List[CToken] = tokenizer.run(content=content)

        res = [token.model_dump(mode="json") for token in tokens]
        if caching_enabled():
            create_caching_file_from_data(
                res, tokenizer.pipeline_step_type, document_id
            )

        return res
