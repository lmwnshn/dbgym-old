import sqlalchemy as sa

from nyoom_flask.extensions import db


class NyoomInstance(db.Model):
    __tablename__ = "nyoom_instance"

    port = sa.Column(sa.Integer, primary_key=True)
    stdin_file = sa.Column(sa.Text, nullable=False)
    stdout_file = sa.Column(sa.Text, nullable=False)
    stderr_file = sa.Column(sa.Text, nullable=False)
    pid = sa.Column(sa.Integer)

    def __repr__(self):
        return (
            f"<{self.__tablename__} "
            f"{self.port=} "
            f"{self.stdin_file=} "
            f"{self.stdout_file=} "
            f"{self.stderr_file=} "
            f"{self.pid=}>"
        )
