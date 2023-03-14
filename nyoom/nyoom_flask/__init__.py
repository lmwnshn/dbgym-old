from flask import Flask
from nyoom_flask.config import Config
from nyoom_flask.extensions import db


def create_app(config=Config):
    app = Flask(__name__)
    app.config.from_object(config)
    app.config["NYOOM_DIR"].mkdir(parents=True, exist_ok=True)
    db.init_app(app)

    from nyoom_flask.model.instance import NyoomInstance

    with app.app_context():
        db.create_all()

    from nyoom_flask.nyoom_flask import nyoom_flask

    app.register_blueprint(nyoom_flask, url_prefix="/nyoom")

    @app.route("/")
    def index():
        return "Borf!"

    return app
