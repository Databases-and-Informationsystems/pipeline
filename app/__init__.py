from flask import Flask
from dotenv import load_dotenv

from app.exception import register_error_handlers


def create_app(config_class):
    app = Flask(__name__)
    load_dotenv()
    app.config.from_object(config_class)

    from .controllers import main

    app.register_blueprint(main)

    register_error_handlers(app)

    return app
