from . import app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import TIMESTAMP as Timestamp

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

# モデルのパラメータのファイル名を保存する
class Model_file(db.Model):
    __tablename__ = 'modelfiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    timestamp = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            text=self.f_name,
            number=self.timestamp
        )

    def __repr__(self):
        return '<model %r>' % self.id


# meta-pathの重みのファイル名を保存する
class Metapath_file(db.Model):
    __tablename__ = 'metapathfiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    timestamp = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            text=self.f_name,
            number=self.timestamp
        )

    def __repr__(self):
        return '<metapath %r, %r, %r>' % self.id


# 潜在表現のファイル名を保存する
class Latent_file(db.Model):
    __tablename__ = 'latentfiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    timestamp = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            text=self.f_name,
            number=self.timestamp
        )

    def __repr__(self):
        return '<metapath %r, %r, %r>' % self.id
