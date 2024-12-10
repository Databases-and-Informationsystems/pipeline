from flask import Flask
from dotenv import load_dotenv

from .routes import blueprint

OPEN_AI_KEY_ENV_CONST = "OPEN_AI_KEY"


def create_app():

    app = Flask(__name__)

    app.register_blueprint(blueprint, url_prefix="/pipeline")

    load_dotenv()

    return app
