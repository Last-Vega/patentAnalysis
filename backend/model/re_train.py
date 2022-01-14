import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import torch
import torch.nn.functional as F
from .args import *
import os
from .util import fix_seed, prepare_adj_for_training, prepare_features_for_training, feedbacked_model_init

fix_seed(42)
def createSampleGraph():
    seed = 42
    n = 100
    p = 0.1
    g = nx.random_graphs.fast_gnp_random_graph(n, p, seed, directed=False)

    adj = np.zeros((n,n))
    edges = g.edges()
    for edge in edges:
        e0 = edge[0]
        e1 = edge[1]
        adj[e0][e1] = 1
        adj[e1][e0] = 1

    adj = csr_matrix(adj)
    features = np.zeros((n, n))
    count = 0
    for elm in features:
        elm[count] += 1
        count += 1

    features = lil_matrix(features)

    return adj, features

def createSampleBiGraph():
    n1 = 100
    n2 = 100
    seed = 42
    p = 0.05
    g = nx.bipartite.generators.random_graph(n1, n2, p, seed)

    n1_set = list(range(0, n1))
    n2_set = list(range(n1, n1+n2))

    edges = g.edges()

    B = nx.Graph()
    B.add_nodes_from(n1_set, bipartite=0) # Add the node attribute "bipartite"
    B.add_nodes_from(n2_set, bipartite=1)

    B.add_edges_from(edges)

    # Separate by group
    # l, r = nx.bipartite.sets(B)
    l = set(range(0,n1))
    r = set(range(n1, n1+n2))

    pos = {}

    # Update position for node from each group
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))

    bi_networks = np.zeros((n1+n2, n1+n2))

    for edge in edges:
        e0 = edge[0]
        e1 = edge[1]
        bi_networks[e0][e1] = 1
        bi_networks[e1][e0] = 1

    bi_networks = csr_matrix(bi_networks)
    return bi_networks

def train(latentC, latentT):
    adj, features = createSampleGraph()
    bi_adj = createSampleBiGraph()
    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(adj)
    features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig, bi_test_edges, bi_test_edges_false = prepare_adj_for_training(bi_adj)
    bipartite_dim = bi_adj.shape[1]

    model, optimizer = feedbacked_model_init(adj_norm, graph_dim, bipartite_dim)

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

    Z = model.Z_t.to('cpu').detach().numpy().copy().tolist()
    return Z