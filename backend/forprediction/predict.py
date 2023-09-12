import os
from typing import List
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
# from .args import *
from .util import *
from .model import HeteroVGAE
from .. import app
prediction_folder = app.config['PREDICTION_FOLDER']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
fix_seed()


graph = load_binary(f'{prediction_folder}/bipartite_graph.pkl')
num_company:int = len({n for n, d in graph.nodes(data=True) if d["bipartite"] == 'company'})
company:list = list(graph.nodes)[:num_company]
term:list = list(graph.nodes)[num_company:]

# /home/watanabe/patentAnalysis/backend/4prediction
# load data
patent_company = load_binary(f'{prediction_folder}/patent_company.pkl')
patent_term = load_binary(f'{prediction_folder}/patent_term.pkl')
cpc = clamp(torch.matmul(patent_company.T, patent_company), 0, 1).to('cpu')
cpt = clamp(torch.matmul(patent_company.T, patent_term), 0, 1).to('cpu')

# def prediction(target_company_list:List)->List[str]:
#     fix_seed()
#     target_company_index:int = [company.index(c) for c in target_company_list]
#     target_company_index.append(company.index('株式会社熊谷組'))
# 	# 熊谷組's index
#     target_company = company.index('株式会社熊谷組')
# 	# prepare for prediction
#     cpc_prediction = create_prediction_tensor(target_company_index, cpc)
#     cpt_prediction = cpt.T

#     feature = create_feature_for_prediction(cpc.shape[1], target_company_index)
#     feature = prepare_features_for_training(feature)
#     input_dim = feature.shape[1]
#     bipartite_dim = cpt_prediction.shape[1]
    
#     cpc_prediction = cpc_prediction.to(device)
#     cpt_prediction = cpt_prediction.to(device)
#     feature = feature.to(device)
    
#     dropout = 0.0
    
#     model = HeteroVGAE(input_dim, bipartite_dim, hidden1_dim=4, hidden2_dim=2, dropout=dropout)
#     model.load_state_dict(torch.load(f'{prediction_folder}/model/model-2.pt'))
#     model.eval()
#     model = model.to(device)
#     z_c, z_t = model.predict(feature, cpc_prediction, cpt_prediction)
    
#     recommended_items = recommendable_items(z_c, z_t, target_company, 100)
#     recommended_items = [term[i] for i in recommended_items]
    
#     return recommended_items

def train(model, optimizer, epoch, feature, adj_norm, bi_adj_norm, adj_label, biadj_label, norm, norm_bi, weight_tensor, weight_tensor_bi, target_company_index):
    history = {'loss':[], 'acc':[], 'ap':[], 'roc':[]}

    model.train()
    for i in range(epoch):
        z_c, z_t, A_pred, Bi_pred = model(feature, adj_norm, bi_adj_norm)

        users_zc = z_c.clone()

        # 28 is kumagaigumi's index
        for i in target_company_index:
            users_zc[i] = z_c[28]
        
        optimizer.zero_grad()

        loss = model.loss_function(norm, norm_bi, adj_label, biadj_label, A_pred, Bi_pred, weight_tensor, weight_tensor_bi, users_zc)

        loss.backward()
        optimizer.step()

        embedding_company = z_c.to('cpu').detach()
        embedding_term = z_t.to('cpu').detach()
    return embedding_company, embedding_term

def prediction(target_company_list:List)->List[str]:
    fix_seed()
    target_company_index:int = [company.index(c) for c in target_company_list]
    target_company_index.append(company.index('株式会社熊谷組'))
	# 熊谷組's index
    target_company = company.index('株式会社熊谷組')
	
    cpc = clamp(torch.matmul(patent_company.T, patent_company), 0, 1).to('cpu')
    cpt = clamp(torch.matmul(patent_company.T, patent_term), 0, 1).to('cpu').T

    adj_norm = load_binary(f'{prediction_folder}/prediction/adj_norm.pkl')
    adj_label = load_binary(f'{prediction_folder}/prediction/adj_label.pkl')
    norm = load_binary(f'{prediction_folder}/prediction/norm.pkl')
    weight_tensor = load_binary(f'{prediction_folder}/prediction/weight_tensor.pkl')
    bi_adj_norm = load_binary(f'{prediction_folder}/prediction/bi_adj_norm.pkl')
    biadj_label = load_binary(f'{prediction_folder}/prediction/biadj_label.pkl')
    norm_bi = load_binary(f'{prediction_folder}/prediction/norm_bi.pkl')
    weight_tensor_bi = load_binary(f'{prediction_folder}/prediction/weight_tensor_bi.pkl')

    feature = create_feature(cpc.shape[1])
    feature = prepare_features_for_training(feature)

    input_dim = feature.shape[1]
    bipartite_dim = cpt.shape[1]

    print(input_dim, bipartite_dim)

    adj_norm = adj_norm.to(device)
    adj_label = adj_label.to(device)
    biadj_label = biadj_label.to(device)
    bi_adj_norm = bi_adj_norm
    feature = feature.to(device)
    
    dropout = 0.0
    
    model = HeteroVGAE(input_dim, bipartite_dim, hidden1_dim=4, hidden2_dim=2, dropout=dropout)
    model = model.to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    z_c, z_t = train(model, optimizer, 200, feature, adj_norm, bi_adj_norm, adj_label, biadj_label, norm, norm_bi, weight_tensor, weight_tensor_bi, target_company_index)
    
    recommended_items_index = recommendable_items(z_c, z_t, target_company, 100)
    recommended_items = [term[i] for i in recommended_items_index]

    z_c = z_c.to('cpu').detach().numpy().copy().tolist()
    z_t = z_t.to('cpu').detach().numpy().copy().tolist()
    
    company_info = []
    # for i in range(len(target_company_index)):
    #     company_info.append({'company':company[target_company_index[i]], 'x':z_c[i][0], 'y':z_c[i][1]})
    for i in target_company_index:
        company_info.append({'company':company[i], 'x':z_c[i][0], 'y':z_c[i][1]})
    term_info = eval(target_company_list, recommended_items, recommended_items_index, z_t)
    return company_info, term_info

def check_term(kumgai_term_index, collaborated_term_index, reccomendable_term_index):
     if reccomendable_term_index in kumgai_term_index and reccomendable_term_index in collaborated_term_index:
         return 't1'
     elif reccomendable_term_index in kumgai_term_index and reccomendable_term_index not in collaborated_term_index: 
        return 't2'
     elif reccomendable_term_index not in kumgai_term_index and reccomendable_term_index in collaborated_term_index:
        return 't3'
     else:
        return 't4'

def eval(target_company_list, recommendable_items, recommended_items_index, z_t):
	# term1 -> kumagaigumi
    target_company = company.index('株式会社熊谷組')
    term1 = [term[i] for i in range(len(term)) if cpt[target_company, i] == 1]
    # term2 -> target_company_list
    term2 = []
    for c in target_company_list:
        term2.extend([term[i] for i in range(len(term)) if cpt[company.index(c), i] == 1])

    term1_index = [i for i in range(len(term)) if term[i] in term1]
    term2_index = [i for i in range(len(term)) if term[i] in term2]

    term_info = []
    # for i in range(len(recommendable_items)):
    #     term_info.append({'term':term[i], 'color':check_term(term1_index, term2_index, recommended_items_index[i]), 'x':z_t[i][0], 'y':z_t[i][1]})
    for i in recommended_items_index:
        term_info.append({'term':term[i], 'color':check_term(term1_index, term2_index, i), 'x':z_t[i][0], 'y':z_t[i][1]})
    return term_info



# def eval(target_company_list, recommendable_items):
# 	# term1 -> kumagaigumi
#     target_company = company.index('株式会社熊谷組')
#     term1 = [term[i] for i in range(len(term)) if cpt[target_company, i] == 1]
#     # term2 -> target_company_list
#     term2 = []
#     for c in target_company_list:
#         term2.extend([term[i] for i in range(len(term)) if cpt[company.index(c), i] == 1])

#     # recommendable_itemsの中で，term1とterm2の両方に含まれる割合を抽出
#     term1_index = [i for i in range(len(term)) if term[i] in term1]
#     term2_index = [i for i in range(len(term)) if term[i] in term2]
#     recommendable_items_index = [i for i in range(len(term)) if term[i] in recommendable_items]
#     term1_and_term2 = [i for i in recommendable_items_index if i in term1_index and i in term2_index]
#     term1_and_term2_term_list = [term[i] for i in term1_and_term2]
#     # cover = len(term1_and_term2) / len(recommendable_items_index)

#     # recommendable_itemsの中で，term1に含まれ，term2に含まれない割合を抽出
#     term1_not_term2 = [i for i in recommendable_items_index if i in term1_index and i not in term2_index]
#     term1_not_term2_term_list = [term[i] for i in term1_not_term2]
#     # cover_not_term2 = len(term1_not_term2) / len(recommendable_items_index)

#     # recommendable_itemsの中で，term1には含まれず，term2に含まれる割合を抽出
#     not_term1_term2 = [i for i in recommendable_items_index if i not in term1_index and i in term2_index]
#     not_term1_term2_term_list = [term[i] for i in not_term1_term2]
#     # cover_not_term1 = len(not_term1_term2) / len(recommendable_items_index)
    
#     # recommendable_itemsの中で，term1には含まれず，term2にも含まれない割合を抽出
#     not_term1_not_term2 = [i for i in recommendable_items_index if i not in term1_index and i not in term2_index]
#     not_term1_not_term2_term_list = [term[i] for i in not_term1_not_term2]
#     # cover_not_term1_not_term2 = len(not_term1_not_term2) / len(recommendable_items_index)

#     return term1_and_term2_term_list, term1_not_term2_term_list, not_term1_term2_term_list, not_term1_not_term2_term_list