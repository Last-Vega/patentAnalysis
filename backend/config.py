import os
class BaseConfig(object):
    DEBUG = True
    TEMP_FOLDER = "/app/backend/temp"
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{password}@{host}/{db_name}'.format(**{
        'user': os.environ.get("DATABASE_USER"),
        'password': os.environ.get("DATABASE_PASSWORD"),
        'host': os.environ.get("DATABASE_HOST"),
        'db_name': os.environ.get("DATABASE_NAME")
    })
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
#     UTILS_FOLDER = "/app/backend/utils"
#     TRAIN_FOLDER = "/app/backend/utils/trainFiles"
#     BIBTEX_FOLDER = "/app/backend/utils/bibtexFiles"