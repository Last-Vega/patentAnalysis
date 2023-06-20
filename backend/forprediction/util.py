from typing import List, Tuple, Dict, Any, Union
import pickle
import torch
import networkx as nx
# from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, vstack, hstack
# import torch_geometric
# from torch_geometric.utils import to_networkx
# from preprocessing import *
# from torch_geometric.data import HeteroData
# from torch_geometric.utils import to_networkx, to_dense_adj, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

def load_binary(f: str)->any:
    """
    バイナリファイルを読み込む
    f: file path
    """
    with open(f, 'rb') as f:
        data = pickle.load(f)
    return data

def save_binary(data: any, f: str)->str:
    """
    バイナリファイルを保存する
    data: 保存するデータ
    f: file path
    """
    with open(f, 'wb') as f:
        pickle.dump(data, f)
    return 'Done'

def fix_seed(seed=42)->None:
    """
    乱数の固定
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return 

# def check_graph(data: torch_geometric.data.Data)->None:
#     '''グラフ情報を表示'''
#     print("グラフ構造:", data)
#     print("グラフのキー: ", data.keys)
#     print("ノード数:", data.num_nodes)
#     print("エッジ数:", data.num_edges)
#     print("ノードの特徴量数:", data.num_node_features)
#     print("孤立したノードの有無:", data.contains_isolated_nodes())
#     print("自己ループの有無:", data.contains_self_loops())
#     print("====== ノードの特徴量:x ======")
#     print(data['x'])
#     print("====== ノードのクラス:y ======")
#     print(data['y'])
#     print("========= エッジ形状 =========")
#     print(data['edge_index'])
#     return None

# def view_graph(data: torch_geometric.data.Data):
#     # networkxのグラフに変換
#     nxg = to_networkx(data)

#     # 可視化のためのページランク計算
#     pr = nx.pagerank(nxg)
#     pr_max = np.array(list(pr.values())).max()

#     # 可視化する際のノード位置
#     draw_pos = nx.spring_layout(nxg, seed=0)

#     # ノードの色設定
#     cmap = plt.get_cmap('tab10')
#     labels = data.y.numpy()
#     colors = [cmap(l) for l in labels]

#     # 図のサイズ
#     plt.figure(figsize=(10, 10))

#     # 描画
#     nx.draw_networkx_nodes(nxg,
#                         draw_pos,
#                         node_size=[v / pr_max * 1000 for v in pr.values()],
#                         node_color=colors, alpha=0.5)
#     nx.draw_networkx_edges(nxg, draw_pos, arrowstyle='-', alpha=0.2)
#     nx.draw_networkx_labels(nxg, draw_pos, font_size=10)

#     plt.title('KarateClub')
#     plt.show()

def clamp(data: torch.Tensor, min: float, max: float)->torch.Tensor:
    """
    行列同士の積だと共起回数になるので，隣接行列の範囲に収めるためにクリッピングする
    data: torch.Tensor
    min: float
    max: float
    """
    return torch.clamp(data, min, max)

def sparse_to_tuple(sparse_mx):
    """
    scipyの疎行列をtorchの疎行列に変換する
    sparse_mx: scipy.sparse.coo_matrix
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    """
    隣接行列を次数行列を用いて正規化する
    adj: scipy.sparse.csr_matrix
    tilda{A} = D^{-1/2}AD^{-1/2} + I の形にする
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def prepare_adj_for_training(adj: csr_matrix)->tuple:
    """
    学習のために，隣接行列を正規化・ラベル付けする
    adj: csr_matrix
    """
    # Store original adjacency matrix (without diagonal entries) for later
    adj_norm = preprocess_graph(adj)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_dense = adj.todense()
    # adj_label = adj_dense
    adj_label = adj
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

    return adj_norm, adj_label, norm, weight_tensor

def prepare_biadj_for_training(adj: csr_matrix)->tuple:
    """
    学習のために，隣接行列を正規化・ラベル付けする
    adj: csr_matrix
    """
    # Store original adjacency matrix (without diagonal entries) for later
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj
    adj_label = sparse_to_tuple(adj_label)

    # print(adj_label[1][adj_label[1] == 1].shape)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2]))
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight

    return adj_label, norm, weight_tensor

def biadj_to_squer_matrix(binary: np.ndarray)->np.ndarray:
    '''二部グラフの隣接行列を正方行列に変換'''
    n = binary.shape[0]
    m = binary.shape[1]
    square = np.zeros((n+m, n+m))
    for i in range(n):
        for j in range(m):
            if binary[i, j] == 1:
                square[i][j+n] = 1
                square[j+n][i] = 1
    return square

def create_feature(feature_dim:int)->lil_matrix:
    """
    特徴量を作成する
    feature_dim: int
    """
    feature = np.eye(feature_dim)
    feature = lil_matrix(feature)
    return feature

def create_feature_tensor(feature_dim:int)->torch.Tensor:
    """
    特徴量を作成する
    feature_dim: int
    """
    feature = torch.eye(feature_dim)
    return feature

def create_feature_for_prediction(feature_dim:int, target_index_list:list)->lil_matrix:
    """
    予測したい企業のインデックスに基づいてのみの特徴量を作成する
    feature_dim: int
    target_index_list: list
    """
    feature = np.zeros((feature_dim, feature_dim))
    for i in target_index_list:
        feature[i, i] = 1
    feature = lil_matrix(feature)
    return feature

def prepare_features_for_training(features:lil_matrix)->torch.sparse.FloatTensor:
    """
    学習のために，特徴量を正規化する
    features: lil_matrix
    """
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                torch.FloatTensor(features[1]),
                                torch.Size(features[2]))
    
    return features

def edge_index_to_list(edge_index:torch.Tensor)->np.ndarray:
    """
    edge_indexをリストに変換する
    edge_index: torch.Tensor
    """
    edge_list = []
    for i in range(edge_index.shape[1]):
        edge_list.append([edge_index[0][i].item(), edge_index[1][i].item()])
    edge_list = np.array(edge_list)
    return edge_list

def edge_index_to_tuple_for_bipartite(edge_index:torch.Tensor, n1:int, n2:int)->np.ndarray:
    """
    edge_indexをリストに変換する
    edge_index: torch.Tensor
    """
    edge_list = []
    for i in range(edge_index.shape[1]):
        edge_list.append((edge_index[0][i].item(), edge_index[1][i].item()+n1))
    
    return edge_list

def make_edge_label(adj:torch.Tensor, edge_index:torch.Tensor)->list:
    """
    edge_indexに対応するラベルを作成する
    adj: torch.Tensor
    edge_index: torch.Tensor
    """
    edge_label = []
    for i in range(len(edge_index)):
        if adj[edge_index[i][0], edge_index[i][1]].item() == 1:
            edge_label.append(1)
        else:
            edge_label.append(0)
    return edge_label


def mask_test_edges(adj)->tuple:
    """
    Function to build test set with 10% positive links
    NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    TODO: Clean up.
    adj: csr_matrix
    """
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_for_bipartite(adj):
    """
    Function to build test set with 10% positive links
    NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    TODO: Clean up.
    adj(bipartite): csr_matrix
    """
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_tuple = sparse_to_tuple(adj)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    

    data = np.ones(train_edges.shape[0])
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def get_scores(emb, adj_orig, edges_pos, edges_neg)->tuple:
    """
    emb: np.ndarray
    adj_orig: csr_matrix
    edges_pos: np.ndarray
    edges_neg: np.ndarray
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label)->float:
    """
    adj_rec: torch.Tensor
    adj_label: torch.Tensor
    """
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def create_prediction_tensor(target_index_list:list, target_tensor:torch.Tensor)->torch.Tensor:
    """
    target_index_list: list of index of target company
    target_tensor: tensor of company-term matrix
    """
    prediction_tensor:torch.Tensor = torch.zeros(target_tensor.shape[0], target_tensor.shape[1])
    for i in target_index_list:
        prediction_tensor[i, :] = target_tensor[i, :]

    return prediction_tensor

def recommendable_items(z_c:torch.Tensor, z_t:torch.Tensor, target_index:int, top_n)->list:
    """
    ユークリッド距離による推薦
    z_c: company latent vector
    z_t: term latent vector
    target_index: index of target company
    top_n: number of recommended items
    """
    target_xy = z_c[target_index, :]
    euclidean_distance = torch.norm(z_t - target_xy, dim=1)
    sorted_index = torch.argsort(euclidean_distance)
    return sorted_index.tolist()[:top_n]