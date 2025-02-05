import traceback

from flask import jsonify, Flask
from flask_restx import ValidationError
from werkzeug.exceptions import HTTPException

from app.util.logger import logger


def register_error_handlers(app: Flask) -> None:

    @app.errorhandler(Exception)
    def handle_global_error(error):
        if isinstance(error, ValidationError):
            status_code = 400  # Bad Request for ValidationError
        elif isinstance(error, HTTPException):
            status_code = error.code  # Use the code from HTTPException
        else:
            status_code = 500  # Default Internal Server Error
        response = {
            "success": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc().splitlines(),
            },
        }

        # Logs always contains trace
        logger.error(str(response))

        # Include traceback only in debug mode
        if not app.config.get("DEBUG", False) and "traceback" in response["error"]:
            del response["error"]["traceback"]

        # Attempt to infer an HTTP status code, defaulting to 500
        return jsonify(response), status_code
