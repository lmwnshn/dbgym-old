import sqlalchemy as sa

from trainer.extensions import db


class TrainerInstance(db.Model):
    __tablename__ = "trainer_instance"

    port = sa.Column(sa.Integer, primary_key=True)
    db_name = sa.Column(sa.Text, nullable=False)
    db_type = sa.Column(sa.Text, nullable=False)
    initialized = sa.Column(sa.Boolean, nullable=False)
    stdin_file = sa.Column(sa.Text, nullable=False)
    stdout_file = sa.Column(sa.Text, nullable=False)
    stderr_file = sa.Column(sa.Text, nullable=False)
    pid = sa.Column(sa.Integer)

    def __repr__(self):
        return (
            f"<{self.__tablename__} "
            f"{self.port=} "
            f"{self.db_name=} "
            f"{self.db_type=} "
            f"{self.initialized=} "
            f"{self.stdin_file=} "
            f"{self.stdout_file=} "
            f"{self.stderr_file=} "
            f"{self.pid=}>"
        )
