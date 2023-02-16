from flask import Flask

from monitor.config import Config
from monitor.extensions import db
from monitor.extensions import scheduler

from sqlalchemy import text


def run_sql(sql):
    with scheduler.app.app_context():
        r = db.session.execute(text(sql))
        # TODO(WAN): logging


def create_app(config=Config):
    app = Flask(__name__)
    app.config.from_object(config)
    db.init_app(app)

    with app.app_context():
        db.create_all()

    scheduler.init_app(app)
    scheduler.start()

    return app
