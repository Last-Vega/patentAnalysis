from . import app
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)

def init_create():
    # db.create_all()
    with app.app_context():
        db.drop_all()
        db.create_all()
    print('done')
    print(db)

class Test(db.Model):
    __tablename__ = 'tests'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(1000))
    number = db.Column(db.Integer)

    def to_dict(self):
        return dict(
            id=self.id,
            text=self.text,
            number=self.number
        )

    def __repr__(self):
        return '<Test %r, %r, %r>' % self.id, self.text, self.number