import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

from .args import *
from .util import *
from .input_data import adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_norm, adj_label, norm, weight_tensor, adj_orig, cpt, bi_train, train_edges_bi, val_edges_bi, val_edges_false_bi, test_edges_bi, test_edges_false_bi, biadj_label, norm_bi, weight_tensor_bi, cpt_tensor
from .model import HeteroVGAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
fix_seed()

print('Start training...')
feature = create_feature(adj_norm.shape[0])
feature = prepare_features_for_training(feature)

input_dim = feature.shape[1]
bipartite_dim = cpt.shape[1]

adj_norm = adj_norm.to(device)
adj_label = adj_label.to(device)
biadj_label = biadj_label.to(device)
cpt_tensor = cpt_tensor.to(device)

# adj_orig = adj_orig.to(device)
weight_tensor = weight_tensor.to(device)
weight_tensor_bi = weight_tensor_bi.to(device)
feature = feature.to(device)
dropout = 0.0
model = HeteroVGAE(input_dim, bipartite_dim, dropout=dropout)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

print(model)

def train(epoch):
    loss_his = []
    acc = []
    ap = []
    roc = []
    model.train()

    for i in range(epoch):
        z_c, z_t, mean, logvar, mu, sigma, A_pred, Bi_pred  = model(feature, adj_norm, cpt_tensor)
        optimizer.zero_grad()

        #TODO define biadj_label
        loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor) + norm_bi*F.binary_cross_entropy(Bi_pred.view(-1), biadj_label.to_dense().view(-1), weight=weight_tensor_bi)
                                                                                             
        
        # kl_divergence = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD1 = -0.5 / A_pred.size(0) * torch.mean(torch.sum(1 + 2 * logvar - mean.pow(2) - logvar.exp().pow(2), 1))
        KLD2 = - 0.5 / Bi_pred.size(0) * torch.mean(torch.sum(1 + 2 * sigma - mu.pow(2) - sigma.exp().pow(2), 1))

        loss = loss + KLD1 + KLD2
        loss.backward()
        optimizer.step()

        embedding_company = z_c.cpu().detach().numpy()
        embedding_term = z_t.cpu().detach().numpy()
        train_acc = get_acc(A_pred, adj_label)
        roc_curr, ap_curr = get_scores(embedding_company, adj_orig, val_edges, val_edges_false)

        loss_his.append(loss.item())
        acc.append(train_acc.item())
        ap.append(ap_curr)
        roc.append(roc_curr)
        print("Epoch:", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_ap=", "{:.5f}".format(ap_curr), 
              "val_roc=", "{:.5f}".format(roc_curr))

    return model, embedding_company, embedding_term, A_pred, loss_his, acc, ap, roc


def test(model, emb, test_edges, test_edges_false):
    model.eval()
    test_roc, test_ap = get_scores(emb, adj_label, test_edges, test_edges_false)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

def test_z(model, test_adj, test_x):
    model.eval()
    test_z = model.encode(test_x, test_adj)
    return test_z



num_train = 1000

model, embedding_company, embedding_term, A_pred, loss_his, acc, ap, roc = train(num_train)
torch.save(model.state_dict(), "/app/backend/4prediction/data/model/model0501.pt")