from flask import Flask, render_template, request, send_file, abort
import os

from routes import routes

def create_app():
    app = Flask(__name__)

    app.config["UPLOAD_FOLDER"] = "static/uploads"
    app.config["OUTPUT_FOLDER"] = "static/outputs"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    app.logger.info("create_app(): registering blueprints")
    app.register_blueprint(routes, url_prefix="/route")

    return app