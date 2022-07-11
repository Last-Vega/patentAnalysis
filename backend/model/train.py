import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, lil_matrix
from .args import *
import os
from .util import fix_seed, prepare_adj_for_training, prepare_features_for_training, model_init
import cupy as cp

# from input_data import adj, features, bi_adj
from .. import app
temp_folder = app.config['TEMP_FOLDER']

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

def weighting(dict, list, size):
    weighted_adj = torch.zeros((size[0],size[1]))
    for weight, original_adj in zip(list, dict.values()):
        original_adj = torch.from_numpy(original_adj).clone().to(torch.float32)
        weighted_adj += weight * original_adj
    weighted_adj = torch.nan_to_num(weighted_adj)
    weighted_adj = weighted_adj.to('cpu').detach().numpy().copy()
    weighted_adj = csr_matrix(weighted_adj)
    return weighted_adj
    

def initialTrain():
    # from .input_data import adj, features, bi_adj
    from .input_data import adj_dict, features, bi_dict
    fix_seed(42)
    # Train on CPU (hide GPU) due to memory constraints
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    adj_shape = adj_dict['CPC'].shape
    bi_shape = bi_dict['CPT'].shape
    adj_weight_list = [1] * len(adj_dict)
    bi_weight_list = [1] * len(bi_dict)
    adj = weighting(adj_dict, adj_weight_list, adj_shape)
    bi_adj = weighting(bi_dict, bi_weight_list, bi_shape)

    weight_tensor, adj_norm, norm, adj_label, adj_orig = prepare_adj_for_training(adj)
    features = prepare_features_for_training(features)
    graph_dim = features.shape[1]

    bi_weight_tensor, bi_adj_norm, bi_norm, bi_adj_label, bi_adj_orig = prepare_adj_for_training(bi_adj)
    bipartite_dim = bi_adj.shape[1]


    model, optimizer = model_init(adj_norm, graph_dim, bipartite_dim)
    print('ok')
    for epoch in range(num_epoch):
        A_pred, Bi_pred = model(features, bi_adj_norm)
        A_pred = A_pred.to('cpu')
        Bi_pred = Bi_pred.to('cpu')
        optimizer.zero_grad()
        # loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor) + bi_norm*F.binary_cross_entropy(Bi_pred.view(-1), bi_adj_label.to_dense().view(-1), weight = bi_weight_tensor)
        # kl_divergence1 = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        # kl_divergence2 = 0.5/ Bi_pred.size(0) * (1 + 2*model.siguma - model.mu**2 - model.siguma**2).sum(1).mean()
        loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor) + bi_norm*F.binary_cross_entropy(Bi_pred.view(-1), bi_adj_label.to_dense().view(-1), weight = bi_weight_tensor)
        kl_divergence1 = 0.5/ A_pred.size(0) * (1 + 2*model.logstd.to('cpu') - model.mean.to('cpu')**2 - torch.exp(model.logstd.to('cpu'))**2).sum(1).mean()
        kl_divergence2 = 0.5/ Bi_pred.size(0) * (1 + 2*model.siguma.to('cpu') - model.mu.to('cpu')**2 - model.siguma.to('cpu')**2).sum(1).mean()
        loss -= kl_divergence1
        loss -= kl_divergence2
        loss.backward()
        optimizer.step()
        print(loss)

    Z = model.Z_t.to('cpu').detach().numpy().copy().tolist()
    Z_c = Z[:num_createdBy]
    Z_t = Z[num_createdBy:]
    import pickle
    with open(f'{temp_folder}/z0304.c', 'wb') as wb:
        pickle.dump(Z_c, wb)

    with open(f'{temp_folder}/z0304.t', 'wb') as wb:
        pickle.dump(Z_t, wb)
