from flask import Flask
from dotenv import load_dotenv

from werkzeug.middleware.proxy_fix import ProxyFix

from .routes import blueprint


def create_app():

    app = Flask(__name__)

    app.register_blueprint(blueprint, url_prefix="/pipeline")

    load_dotenv()

    return app
