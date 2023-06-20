from . import db
from .table import *
# coding: utf-8


def seed():
    # テーブルのデータを消去
    db.session.query(Test).delete()

    # 初期データを追加
    test_list = [
        Test(
            id=1,
            text="text_1",
            number=10
        ),
        Test(
            id=2,
            text="text_2",
            number=100
        )
    ]

    db.session.add_all(test_list)
    db.session.commit()
