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
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<model %r, %r>' % self.id, self.f_name


# meta-pathの重みのファイル名を保存する
class Metapath_Adj_file(db.Model):
    __tablename__ = 'adjfiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<adj %r, %r>' % self.id, self.f_name

# meta-pathの重みのファイル名を保存する
class Metapath_Bi_file(db.Model):
    __tablename__ = 'bifiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<bi %r, %r>' % self.id, self.f_name

# meta-pathの重みのファイル名を保存する
class company_criteria_file(db.Model):
    __tablename__ = 'companycriteria'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<adj %r, %r>' % self.id, self.f_name

# meta-pathの重みのファイル名を保存する
class term_criteria_file(db.Model):
    __tablename__ = 'term_criteria'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<bi %r, %r>' % self.id, self.f_name


# 潜在表現のファイル名を保存する
class Latent_company_file(db.Model):
    __tablename__ = 'companyfiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<company %r, %r>' % self.id, self.f_name


class Latent_term_file(db.Model):
    __tablename__ = 'termfiles'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    f_name = db.Column(db.String(1000))
    created_at = db.Column(Timestamp)

    def to_dict(self):
        return dict(
            id=self.id,
            f_name=self.f_name,
            created_at=self.created_at
        )

    def __repr__(self):
        return '<term %r, %r>' % self.id, self.f_name
