# from unittest import result
from flask import Blueprint, jsonify, request
from .table import *
import torch
import numpy as np
from .model.re_train import train, recommend, vstrain
from .model.train import initialTrain
from .util import calcCCDistance, calcCTDistance, calcTTDistance, appendElm, calcCTDistanceForRecommend, loadBinary, appendElmForRecommend, loadBinary
# coding: utf-8
from . import app
temp_folder = app.config['TEMP_FOLDER']
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
    adj = loadBinary(f'{temp_folder}/W_adj0122.adj')
    result = {'adj': adj.toarray().tolist()}
    result = jsonify(result)
    return result
    