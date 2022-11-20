from flask import Flask
from dbgym.trainer import BUILD_DIR
from dbgym.trainer.postgres import postgres


app = Flask(__name__)
app.register_blueprint(postgres, url_prefix="/postgres")
app.config["BUILD_DIR"] = BUILD_DIR


@app.route("/")
def index():
    links = "".join([f"<li>{str(rule)}</li>" for rule in app.url_map.iter_rules()])
    output = f"<p><ul>{links}</ul></p>"
    return output
