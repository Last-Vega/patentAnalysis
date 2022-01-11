from flask import Flask, render_template
from flask_cors import CORS
import datetime

# Flaskの設定
app = Flask(__name__,
            static_folder="../dist/static",
            template_folder="../dist")

# app.config.from_object('backend.config.BaseConfig')

# from .db_schema import db
# db.drop_all()
# db.create_all()

# from .seedings import seed
# seed()

from flask_jwt_extended import JWTManager
app.config['JWT_SECRET_KEY'] = 'seecret'
app.config['JWT_ALGORITHM'] = 'HS256'
app.config['JWT_EXPIRATION_DELTA'] = datetime.timedelta(days=7)
app.config['JWT_NOT_BEFORE_DELTA'] = datetime.timedelta(seconds=0)
jwt = JWTManager(app)

# api.pyで記述したapiへのリクエストURLを登録
from .api import api
app.register_blueprint(api, url_prefix="/api")
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


# 全てのリクエストをVueに投げる。
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return render_template("index.html")