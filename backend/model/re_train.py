import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn.functional as F
from .args import *
import os
import pickle
from .util import fix_seed, prepare_adj_for_training, prepare_features_for_training, feedbacked_model_init, e_step, m_step, model_init, criteria
from .. import app
temp_folder = app.config['TEMP_FOLDER']


def savePickle(f, data):
    with open(f'{temp_folder}/{f}', 'wb') as wf:
        pickle.dump(data, wf)
    return 

def train(latentC, latentT, uCI, uTI):
    fix_seed(42)
    from .input_data import adj, bi_adj, features, adj_dict, bi_dict

    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(adj)
    features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig, bi_test_edges, bi_test_edges_false = prepare_adj_for_training(bi_adj)
    bipartite_dim = bi_adj.shape[1]

    model, optimizer = feedbacked_model_init(adj_norm, graph_dim, bipartite_dim)
    adj_dict = criteria(adj_dict, uCI, latentC)
    bi_dict = criteria(bi_dict, uTI, latentT)
    for e in range(2):
        A_pred, Bi_pred = model(features, bi_adj_norm, latentC, latentT)
        weighted_adj, weight_list = e_step(adj_dict, A_pred)
        weighted_bi, weight_list_bi = e_step(bi_dict, Bi_pred)
        Z_c, Z_t, model, optimizer = m_step(model, optimizer, weighted_adj, features, weighted_bi, latentC, latentT)
    
    # x = random.randint(0,100)
    # print(x)
    # torch.seed(x)
    savePickle('weightAdj.list', weight_list)
    savePickle('weightBi.list', weight_list_bi)
    max_cc = list(adj_dict.keys())[weight_list.index(max(weight_list))]
    max_ct = list(bi_dict.keys())[weight_list_bi.index(max(weight_list_bi))]
    return Z_c, Z_t, max_cc, max_ct

def loadBinary(f_name):
    with open(f_name, 'rb') as rb:
        data = pickle.load(rb)
    return data

def weighting(dict, list, size):
    weighted_adj = torch.zeros((size[0],size[1]))
    for weight, original_adj in zip(list, dict.values()):
        original_adj = torch.from_numpy(original_adj).clone().to(torch.float32)
        weighted_adj += weight * original_adj
    weighted_adj = torch.nan_to_num(weighted_adj)
    weighted_adj = weighted_adj.to('cpu').detach().numpy().copy()
    weighted_adj = csr_matrix(weighted_adj)
    return weighted_adj


def createFeatures(adj:csr_matrix) -> lil_matrix:
    n = adj.shape[0]

    features = np.zeros((n, n))
    for i in range(n):
        features[i][i] = 1
    features = lil_matrix(features)
    return features

def recommend():
    fix_seed(42)
    adj_dict = loadBinary(f'{temp_folder}/adj0213.dict')
    bi_dict = loadBinary(f'{temp_folder}/bi0213.dict')
    adj_weight_list = loadBinary(f'{temp_folder}/weightAdj.list')
    print(adj_weight_list)
    bi_weight_list = loadBinary(f'{temp_folder}/weightBi.list')
    adj_shape = adj_dict['CPC'].shape
    bi_shape = bi_dict['CPT'].shape
    weighted_adj = weighting(adj_dict, adj_weight_list, adj_shape)
    weighted_bi = weighting(bi_dict, bi_weight_list, bi_shape)

    features = createFeatures(weighted_adj)
    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(weighted_adj)
    features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig, bi_test_edges, bi_test_edges_false = prepare_adj_for_training(weighted_bi)
    bipartite_dim = weighted_bi.shape[1]

    model, optimizer = model_init(adj_norm, graph_dim, bipartite_dim)

    for epoch in range(2):
        A_pred, Bi_pred = model(features, bi_adj_norm)
        optimizer.zero_grad()
        loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor) + bi_norm*F.binary_cross_entropy(Bi_pred.view(-1), bi_adj_label.to_dense().view(-1), weight = bi_weight_tensor)
        kl_divergence1 = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        kl_divergence2 = 0.5/ Bi_pred.size(0) * (1 + 2*model.siguma - model.mu**2 - model.siguma**2).sum(1).mean()
        loss -= kl_divergence1
        loss -= kl_divergence2
        loss.backward()
        optimizer.step()
        print(loss)
    
    Z = model.Z_t.to('cpu').detach().numpy().copy().tolist()
    Z_c = Z[:graph_dim]
    Z_t = Z[graph_dim:]
    return Z_c, Z_t


def vstrain(latentC, latentT, uCI, uTI):
    from .input_data import forCompare

    adj_dict, bi_dict, features = forCompare()
    adj_weight_list = loadBinary(f'{temp_folder}/vsCC0307.w')
    bi_weight_list = loadBinary(f'{temp_folder}/vsCT0307.w')
    adj_shape = adj_dict['CPC'].shape
    bi_shape = bi_dict['CPT'].shape
    adj = weighting(adj_dict, adj_weight_list, adj_shape)
    bi_adj = weighting(bi_dict, bi_weight_list, bi_shape)

    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(adj)
    features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig, bi_test_edges, bi_test_edges_false = prepare_adj_for_training(bi_adj)
    bipartite_dim = bi_adj.shape[1]

    model, optimizer = feedbacked_model_init(adj_norm, graph_dim, bipartite_dim)
    adj_dict = criteria(adj_dict, uCI, latentC)
    bi_dict = criteria(bi_dict, uTI, latentT)
    for e in range(2):
        A_pred, Bi_pred = model(features, bi_adj_norm, latentC, latentT)
        weighted_adj, weight_list = e_step(adj_dict, A_pred)
        weighted_bi, weight_list_bi = e_step(bi_dict, Bi_pred)
        Z_c, Z_t, model, optimizer = m_step(model, optimizer, weighted_adj, features, weighted_bi, latentC, latentT)
        

    savePickle('vsCC0307.w', weight_list)
    savePickle('vsCT0307.w', weight_list_bi)
    max_cc = list(adj_dict.keys())[weight_list.index(max(weight_list))]
    max_ct = list(bi_dict.keys())[weight_list_bi.index(max(weight_list_bi))]
    return Z_c, Z_t, max_cc, max_ct