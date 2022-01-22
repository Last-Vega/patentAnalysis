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

fix_seed(42)

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
        

    # Z = model.Z_t.to('cpu').detach().numpy().copy().tolist()
    # z_c, z_t = model.prediction(features, bi_adj_norm)

    # Z = model.prediction(features, bi_adj_norm).to('cpu').detach().numpy().copy().tolist()
    # Z_c = Z[:graph_dim]
    # Z_t = Z[graph_dim:]
    return Z_c, Z_t


    """
    W = torch.rand(len(metapath_list))

    weighted_adj = torch.zeros((50,50))

    for adj_elm, w in zip(adj_dict.values(), W):
        adj_elm = torch.from_numpy(adj_elm).clone().to(torch.float32)
        weighted_adj += w * adj_elm
    
    weighted_adj = csr_matrix(weighted_adj)


    for epoch in range(num_epoch):
        A_pred, Bi_pred = model(features, bi_adj_norm, latentC, latentT)
        optimizer.zero_grad()
        loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor) + bi_norm*F.binary_cross_entropy(Bi_pred.view(-1), bi_adj_label.to_dense().view(-1), weight = bi_weight_tensor)
        kl_divergence1 = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        kl_divergence2 = 0.5/ Bi_pred.size(0) * (1 + 2*model.siguma - model.mu**2 - model.siguma**2).sum(1).mean()
        loss -= kl_divergence1
        loss -= kl_divergence2
        loss.backward()
        optimizer.step()
        print(loss)

    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(weighted_adj)
    model, optimizer = feedbacked_model_init(adj_norm, graph_dim, bipartite_dim)
    for e in range(10):
        A_pred = model(features)
        weighted_adj, weight_list = e_step(adj_dict, A_pred)
        z, model, optimizer = m_step(model, optimizer, weighted_adj, features, args.modelType)
        adj_dict = check_distance(z, adj_dict, neighborhood_dict)

    """