# from unittest import result
from flask import Blueprint, jsonify, request
from .table import *
from sqlalchemy import desc
import torch
import numpy as np
from .model.re_train import train, recommend, vstrain
from .model.train import initialTrain
from .util import calcCCDistance, calcCTDistance, calcTTDistance, appendElm, calcCTDistanceForRecommend, loadBinary, appendElmForRecommend, loadBinary
from .forprediction.predict import *
# coding: utf-8
from . import app
temp_folder = app.config['TEMP_FOLDER']
prediction_folder = app.config['PREDICTION_FOLDER']
api = Blueprint('api', __name__)

# @api.route('/initial', methods=['POST'])
# def initial():
#     m = request.get_json()['m']
#     print(m)
#     # initialTrain()
#     result = {'message': 'success'}
#     s = initialTrain()
#     return jsonify(result)

@api.route('/test', methods=['POST'])
def api_test():
    return 'success'


@api.route('/test_db', methods=['POST'])
def api_testdb():
    # test_list = db.session.query(Test).all()
    # test_dict = [test.to_dict() for test in test_list]
    # return jsonify(test_dict)
    result = {'message': str(type(db))}
    return jsonify(result)

@api.route('/update', methods=['POST'])
def update():
    latent_c = request.get_json()['companyZ']
    latent_t = request.get_json()['termZ']
    connected = []
    connected += latent_c
    connected += latent_t

    updateCompanyIndex = request.get_json()['CompanyIndex']
    updateTermIndex = request.get_json()['TermIndex']
    updateTermIndex = [index+50 for index in updateTermIndex]
    updateTermIndex += updateCompanyIndex

    tensor_latentC = torch.tensor(latent_c, requires_grad=True, dtype=torch.float32)
    tensor_latentT = torch.tensor(connected, requires_grad=True, dtype=torch.float32)
    # tensor_latentC = torch.FloatTensor(latent_c)
    # tensor_latentT = torch.FloatTensor(connected)

    Z_c, Z_t, maxCCPath, maxCTPath = train(tensor_latentC, tensor_latentT, updateCompanyIndex, updateTermIndex)

    result = {'company': Z_c, 'term': Z_t, 'maxCCPath': maxCCPath, 'maxCTPath': maxCTPath}
    result = jsonify(result)
    return result

@api.route('/search', methods=['POST'])
def search():
    latent_c = request.get_json()['companyZ']
    latent_t = request.get_json()['termZ']
    query = request.get_json()['query']
    company_name = request.get_json()['company']
    term = request.get_json()['term']

    flag = -1
    query_index = -1
    if query in company_name:
        flag = 0
        query_index = company_name.index(query)
    elif query in term:
        flag = 1
        query_index = term.index(query)
    else:
        message = '別の検索語を入力してください'
        result = {'message': message, 'showFlag': False}
        result = jsonify(result)
        return result

    if flag == 0:
        XY = latent_c[query_index]
        close_company_index = calcCCDistance(latent_c, query_index)
        close_term_index = calcCTDistance(latent_c, latent_t, query_index)
        close_company = appendElm(close_company_index, company_name)
        close_term = appendElm(close_term_index, term)
    elif flag == 1:
        XY = latent_t[query_index]
        close_company_index = calcCTDistance(latent_c, latent_t, query_index)
        close_term_index = calcTTDistance(latent_t, query_index)
        close_company = appendElm(close_company_index, company_name)
        close_term = appendElm(close_term_index, term)

    result = {'showFlag': True, 'closeComapny': close_company, 'closeTerm': close_term, 'XY': XY}
    result = jsonify(result)
    return result


@api.route('/recommend', methods=['POST'])
def recommendation():
    Z_c, Z_t = recommend()
    close_term_index = calcCTDistanceForRecommend(Z_c, Z_t, 11)
    term_list = loadBinary(f'{temp_folder}/term.termlist')
    close_term = appendElmForRecommend(close_term_index, term_list)
    result = {'closeTerm': close_term, 'length': len(close_term)}
    return jsonify(result)


@api.route('/vsupdate', methods=['POST'])
def vsupdate():
    latent_c = request.get_json()['companyZ']
    latent_t = request.get_json()['termZ']
    connected = []
    connected += latent_c
    connected += latent_t
    updateCompanyIndex = request.get_json()['CompanyIndex']
    updateTermIndex = request.get_json()['TermIndex']
    updateTermIndex = [index+50 for index in updateTermIndex]
    updateTermIndex += updateCompanyIndex
    
    tensor_latentC = torch.tensor(latent_c, requires_grad=True, dtype=torch.float32)
    tensor_latentT = torch.tensor(connected, requires_grad=True, dtype=torch.float32)

    Z_c, Z_t, maxCCPath, maxCTPath = vstrain(tensor_latentC, tensor_latentT, updateCompanyIndex, updateTermIndex)

    result = {'company': Z_c, 'term': Z_t, 'maxCCPath': maxCCPath, 'maxCTPath': maxCTPath}
    result = jsonify(result)
    return result

@api.route('/printadj', methods=['POST'])
def print_adj():
    # adj = loadBinary(f'{temp_folder}/W_adj1213.adj')
    # result = {'adj': adj.toarray().tolist()}
    # result = jsonify(result)
    # adj = loadBinary(f'{temp_folder}/adj0123.dict')
    # result = {'adj': len(adj['CPC'].tolist())}
    # result = jsonify(result)
    adj = loadBinary(f'{temp_folder}/bi0122-1.dict')
    result = {'adj': len(adj['CPT'].tolist())}
    result = jsonify(result)
    return result


@api.route('/latent', methods=['POST'])
def view_new_latent():
    if db.session.query(Latent_company_file.f_name).order_by(desc(Latent_company_file.created_at)).all():
        company_f = db.session.query(Latent_company_file.f_name).order_by(
            desc(Latent_company_file.created_at)).all()[0][0]
        company_z = loadBinary(f'{temp_folder}/{company_f}')
    else:
        result = {'flag':False}
        return jsonify(result)
    if db.session.query(Latent_term_file.f_name).order_by(desc(Latent_term_file.created_at)).all():
        term_f = db.session.query(Latent_term_file.f_name).order_by(
            desc(Latent_term_file.created_at)).all()[0][0]
        term_z = loadBinary(f'{temp_folder}/{term_f}')
    else:
        result = {'flag': False}
        return jsonify(result)
    

    company_info = {'key': []}
    company = ['株式会社大林組', '大成建設株式会社', '佐藤工業株式会社', '三井住友建設株式会社', '株式会社竹中工務店', '株式会社安藤・間', '日立建機株式会社', '鹿島建設株式会社', '川崎重工業株式会社', '株式会社奥村組', '戸田建設株式会社', 'ジオスター株式会社', '株式会社熊谷組', '五洋建設株式会社', '清水建設株式会社', '株式会社フジタ', 'ＪＩＭテクノロジー株式会社', '西松建設株式会社', 'オリエンタル白石株式会社', '株式会社竹中土木', '飛島建設株式会社', '岐阜工業株式会社', '前田建設工業株式会社', '三菱重工業株式会社', '大豊建設株式会社', '株式会社不動テトラ', '株式会社小松製作所', '東洋建設株式会社', '日立造船株式会社', '株式会社日建設計', '東亜建設工業株式会社', '東急建設株式会社', '成和リニューアルワークス株式会社', 'ＫＹＢ株式会社', '日本製鉄株式会社', 'オイレス工業株式会社', '高砂熱学工業株式会社', '株式会社技研製作所', '岡部株式会社', '株式会社ＩＨＩ建材工業', '東日本高速道路株式会社', '株式会社ガイアート', '東海旅客鉄道株式会社', '株式会社長谷工コーポレーション', 'ＪＲ東日本コンサルタンツ株式会社', '鉄建建設株式会社', '公益財団法人鉄道総合技術研究所', '東京電力ホールディングス株式会社', '北陸鋼産株式会社', 'テクノプロ株式会社']

    for c, xy in zip(company, company_z):
        child = {}
        child['company'] = c
        child['x'] = xy[0]
        child['y'] = xy[1]
        company_info['key'].append(child)
    
    term = ['駆動モータ', '鋼殻', 'シールドジャッキ', 'トンネル掘削', '掘削土砂', '延設', '連設', '免震装置', '支持構造', '土圧', '防水シート', 'スキンプレート', '揺動', 'シール部材', '補強部材', '切断装置', 'シールド掘削', 'トンネル坑内', '水壁', '水平部材', '鉄骨梁', '弾性部材', 'プレストレス', '鋼棒', '継手構造', '免震', '回動自在', 'アンカーボルト', 'ガイドレール', '給気', '既設トンネル', 'エレクタ装置', '進方向', '係合溝', '掘削機本体', '油圧ジャッキ', '台車本体', '工コンクリート', 'ガイド部材', '排泥', '固定部材', '軟弱地盤', 'シールドトンネル', 'トンネル軸方向', '止水性', '連結構造', 'トンネル構造', '止水', 'セグメントピース', 'プレート部材', 'セグメントリング', 'シールド本体', '耐震補強', '天端', '掘削ズリ', '建物躯', '浄化装置', '梁部材', 'コンクリート部材', 'アンカー部材', '柱部材', '搬送台車', '圧縮空気', '挟持', '支保工', '移動台車', '耐震補強構造', 'ベースプレート', '継手部材', '補強構造', '掘削装置', '昇降装置', 'ボルト孔', '挿通孔', '支持ブラケット', '延設方向', '水シート', '接続部材', '接合部材', '支持装置', '油圧駆動装置', '掘削孔', '開孔', '妻型', 'ボルト挿通孔', '止水板', '地下躯', '噴射ノズル', '鋼管杭', '上面開口', '斜行ハニカム', 'コルゲート状シート', '空気供給', '筒状', '接合構造', '橋軸方向', '棒状部材', '伸縮装置', '野縁', 'プレストレス導入']

    term_info = {'key': []}
    for t, xy in zip(term, term_z):
        child = {}
        child['term'] = t
        child['x'] = xy[0]
        child['y'] = xy[1]
        term_info['key'].append(child)

    result = {'companyInfo':company_info, 'termInfo':term_info, 'flag':True}
    return jsonify(result)


@api.route('/predict', methods=['POST'])
def predict_app():
    """
    input: company_name, term_name
    output: recommendable items
    """
    company_name:list = request.get_json()['company']
    # term:list = request.get_json()['term']
    recommendable_items = prediction(company_name)
    result = {'recommendable_items': recommendable_items}

    return jsonify(result)
    # return 'success'

@api.route('/getCompanyName', methods=['POST'])
def get_company_name():
    company_node = loadBinary(f'{prediction_folder}/company_node.pkl')
    company_list = []
    for node in company_node:
        if node == '株式会社熊谷組':
            continue
        else:
            _ = {}
            _['company'] = node
            company_list.append(_)
    result = {'companyList': company_list}
    return jsonify(result)


@api.route('/deldb', methods=['POST'])
def del_db():
    db.drop_all()
    db.create_all()
    return 'success'

@api.route('/debug', methods=['POST'])
def debug():
    from .model.input_data import f_name, adj

    result = {'f_name': f_name, 'adj': str(type(adj))}
    return jsonify(result)