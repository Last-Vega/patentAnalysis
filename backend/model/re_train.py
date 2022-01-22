import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn.functional as F
from .args import *
import os
import pickle
from .util import fix_seed, prepare_adj_for_training, prepare_features_for_training, feedbacked_model_init, e_step, m_step
from .. import app
temp_folder = app.config['TEMP_FOLDER']

fix_seed(42)

def savePickle(f, data):
    with open(f'{temp_folder}/{f}', 'wb') as wf:
        pickle.dump(data, wf)
    return 

def train(latentC, latentT):
    from .input_data import adj, bi_adj, features, adj_dict, bi_dict

    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(adj)
    features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig, bi_test_edges, bi_test_edges_false = prepare_adj_for_training(bi_adj)
    bipartite_dim = bi_adj.shape[1]

    model, optimizer = feedbacked_model_init(adj_norm, graph_dim, bipartite_dim)
    
    for e in range(10):
        A_pred, Bi_pred = model(features, bi_adj_norm, latentC, latentT)
        weighted_adj, weight_list = e_step(adj_dict, A_pred)
        weighted_bi, weight_list_bi = e_step(bi_dict, Bi_pred)
        Z_c, Z_t, model, optimizer = m_step(model, optimizer, weighted_adj, features, weighted_bi, latentC, latentT)
        

    savePickle('weightAdj.list', weight_list)
    savePickle('weightBi.list', weight_list_bi)
    return Z_c, Z_t
