from .util import *
import torch
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from typing import Union, Tuple
# import torch_geometric.transforms as T
# from torch_geometric.data import Data, HeteroData
# from torch_geometric.utils import to_networkx, to_dense_adj, negative_sampling
from .. import app
prediction_folder = app.config['PREDICTION_FOLDER']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fix_seed()

"""
patent_*** = (num_patent, num_***) num_patent = 44564
"""

patent_company = load_binary(f'{prediction_folder}/patent_company.pkl')
patent_term = load_binary(f'{prediction_folder}/patent_term.pkl')
cpc = clamp(torch.matmul(patent_company.T, patent_company), 0, 1).to('cpu')
cpc = csr_matrix(cpc)
adj_orig = cpc

# prepare for training
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(cpc)
adj_norm, adj_label, norm, weight_tensor = prepare_adj_for_training(adj_train)

cpt = clamp(torch.matmul(patent_company.T, patent_term), 0, 1).to('cpu').T
cpt_tensor = cpt
cpt = csr_matrix(cpt)
bi_train, train_edges_bi, val_edges_bi, val_edges_false_bi, test_edges_bi, test_edges_false_bi = mask_test_edges_for_bipartite(cpt)

biadj_label, norm_bi, weight_tensor_bi = prepare_biadj_for_training(bi_train)


