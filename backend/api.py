from flask import Blueprint, jsonify, request
import torch
import numpy as np
from .model.re_train import train
from .util import calcCCDistance, calcCTDistance, calcTTDistance, appendElm
# coding: utf-8

api = Blueprint('api', __name__)


@api.route('/hoge/<string:str>', methods=["GET"])
def hogeGet(str):
    return "hogeGet: " + str


@api.route("/hoge", methods=["POST"])
def hogePost():
    text = request.form["text"]
    return "hogePost: " + text

@api.route('/update', methods=['POST'])
def update():
    latent_c = request.get_json()['companyZ']
    latent_t = request.get_json()['termZ']
    connected = []
    connected += latent_c
    connected += latent_t

    tensor_latentC = torch.tensor(latent_c, requires_grad=True, dtype=torch.float32)
    tensor_latentT = torch.tensor(connected, requires_grad=True, dtype=torch.float32)

    Z_c, Z_t = train(tensor_latentC, tensor_latentT)
    # Z_c = Z[:100]
    # Z_t = Z[100:]
    # print(Z_c[0])
    result = {'company': Z_c, 'term': Z_t}
    result = jsonify(result)
    # result = {'company': latent_c}
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
        close_company_index = calcCCDistance(latent_c, query_index)
        close_term_index = calcCTDistance(latent_c, latent_t, query_index)
        close_company = appendElm(close_company_index, company_name)
        close_term = appendElm(close_term_index, term)
    elif flag == 1:
        close_company_index = calcCTDistance(latent_c, latent_t, query_index)
        close_term_index = calcTTDistance(latent_t, query_index)
        close_company = appendElm(close_company_index, company_name)
        close_term = appendElm(close_term_index, term)

    print(close_company)
    result = {'showFlag': True, 'closeComapny': close_company, 'closeTerm': close_term}
    result = jsonify(result)
    return result