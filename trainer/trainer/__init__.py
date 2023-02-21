from flask import Flask

from trainer.config import Config
from trainer.extensions import db


def create_app(config=Config):
    app = Flask(__name__)
    app.config.from_object(config)
    app.config["TRAINER_DIR"].mkdir(parents=True, exist_ok=True)
    db.init_app(app)

    from trainer.model.instance import Instance

    with app.app_context():
        db.create_all()

    from trainer.postgres import postgres

    app.register_blueprint(postgres, url_prefix="/postgres")

    @app.route("/")
    def index():
        return "Borf!"

    return app
