from scipy.sparse.csr import csr_matrix
from scipy.sparse.lil import lil_matrix
import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
from scipy.spatial import distance
from .preprocessing import *

from .args import *
from .model import Recommendation, RecommendViaFeedback

# 乱数シード固定（再現性の担保）
def fix_seed(seed):
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_adj_for_training(adj:csr_matrix):
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                torch.FloatTensor(adj_norm[1]),
                                torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                torch.FloatTensor(adj_label[1]),
                                torch.Size(adj_label[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false

def prepare_features_for_training(features:lil_matrix):
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                torch.FloatTensor(features[1]),
                                torch.Size(features[2]))
    
    return features

def model_init(adj_norm, graph_dim, bipartite_dim):
    model = Recommendation(adj_norm, graph_dim, bipartite_dim)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

def feedbacked_model_init(adj_norm, graph_dim, bipartite_dim):
    model = RecommendViaFeedback(adj_norm, graph_dim, bipartite_dim)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

def calc_log_likelihood(adj, pred):
    adj_pt = torch.from_numpy(adj).clone().to(torch.float32)
    log_likelihood = torch.matmul(adj_pt, torch.log(pred))
    log_likelihood = torch.sum(log_likelihood)
    return log_likelihood

def calc_metapath_weight(contribution_rate):
    weight_list = []
    rate_sum = sum(contribution_rate)
    for elm in contribution_rate:
        weight = elm/rate_sum
        weight_list.append(weight)
    return weight_list

def feedback(adj_dict, update_index_list, G, bib_database, neighborhood_dict, metapath_list, argument):
    adj_dict['nothing'] = np.zeros((50, 50))
    for update_index in update_index_list:
        update_dict = {update_index: neighborhood_dict[update_index]}
        # adj_dict = reccomend_criteria(G, update_dict, adj_dict, bib_database, metapath_list, argument)
    
    return adj_dict

def e_step(adj_dict, pred):
    contribution_rate = []
    size = 0
    for key, original_adj in adj_dict.items():
        log_liklihood = calc_log_likelihood(original_adj, pred)
        contribution_rate.append(log_liklihood)
        size = original_adj.shape

    weight_list = calc_metapath_weight(contribution_rate)
    print(weight_list)
    # weighted_adj = torch.zeros((50,50))
    weighted_adj = torch.zeros((size[0],size[1]))
    for weight, original_adj in zip(weight_list, adj_dict.values()):
        original_adj = torch.from_numpy(original_adj).clone().to(torch.float32)
        weighted_adj += weight * original_adj
    weighted_adj = torch.nan_to_num(weighted_adj)
    weighted_adj = weighted_adj.to('cpu').detach().numpy().copy()
    weighted_adj = sp.csr_matrix(weighted_adj)
    return weighted_adj, weight_list

def m_step(model, optimizer, adj, features, bi_adj, latentC, latentT):
    weight_tensor, adj_norm, norm, adj_label, adj_orig, test_edges, test_edges_false = prepare_adj_for_training(adj)
    # features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig, bi_test_edges, bi_test_edges_false = prepare_adj_for_training(bi_adj)
    bipartite_dim = bi_adj.shape[1]

    model, optimizer = feedbacked_model_init(adj_norm, graph_dim, bipartite_dim)
    model.train()
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

    model.eval()
   
    Z = model.prediction(features, bi_adj_norm).to('cpu').detach().numpy().copy().tolist()
    Z_c = Z[:graph_dim]
    Z_t = Z[graph_dim:]

    return Z_c, Z_t, model, optimizer

def criteria(adj_dict, updateList, latentData):
    K = 5
    latentData = latentData.detach().numpy()
    pri_dist = distance.cdist(latentData, latentData, metric='euclidean')
    tri_dist = distance.cdist(latentData, latentData, metric='euclidean')
    for updateIndex in updateList:
        closeIndexList = np.argpartition(tri_dist[updateIndex], K)[:K-len(tri_dist[updateIndex])].tolist()
        for closeIndex in closeIndexList:
            for key, adjOrig in adj_dict.items():
                adj_dict[key][closeIndex] += 1
    return adj_dict