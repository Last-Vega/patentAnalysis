from flask import Blueprint, jsonify, request
import torch
from .model.re_train import train
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
    tensor_latentC = torch.tensor(latent_c, requires_grad=True, dtype=torch.float32)
    tensor_latentT = torch.tensor(latent_t, requires_grad=True, dtype=torch.float32)
    # print(f'initial C is {tensor_latentC.dtype}')
    # print(f'initial T is {tensor_latentT.dtype}')
    Z = train(tensor_latentC, tensor_latentT)
    Z_c = Z[:100]
    Z_t = Z[100:]
    result = {'company': Z_c, 'term': Z_t}
    return result