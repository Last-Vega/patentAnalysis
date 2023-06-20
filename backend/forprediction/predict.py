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

def prediction(target_company_list:List)->List[str]:
    fix_seed()
    target_company_index:int = [company.index(c) for c in target_company_list]
    target_company_index.append(company.index('株式会社熊谷組'))
	# 熊谷組's index
    target_company = company.index('株式会社熊谷組')
	# prepare for prediction
    cpc_prediction = create_prediction_tensor(target_company_index, cpc)
	# cpt_prediction = create_prediction_tensor(target_company_index, cpt).T
	# cpc_prediction = cpc
    cpt_prediction = cpt.T

    feature = create_feature_for_prediction(cpc.shape[1], target_company_index)
    feature = prepare_features_for_training(feature)
    input_dim = feature.shape[1]
    bipartite_dim = cpt_prediction.shape[1]
    
    cpc_prediction = cpc_prediction.to(device)
    cpt_prediction = cpt_prediction.to(device)
    feature = feature.to(device)
    
    dropout = 0.0
    
    model = HeteroVGAE(input_dim, bipartite_dim, hidden1_dim=4, hidden2_dim=2, dropout=dropout)
    model.load_state_dict(torch.load(f'{prediction_folder}/model/model-2.pt'))
    model.eval()
    model = model.to(device)
    z_c, z_t = model.predict(feature, cpc_prediction, cpt_prediction)
    
    recommended_items = recommendable_items(z_c, z_t, target_company, 100)
    recommended_items = [term[i] for i in recommended_items]
    
    return recommended_items

def eval(target_company_list, recommendable_items):
	# term1 -> kumagaigumi
    target_company = company.index('株式会社熊谷組')
    term1 = [term[i] for i in range(len(term)) if cpt[target_company, i] == 1]
    # term2 -> target_company_list
    term2 = []
    for c in target_company_list:
        term2.extend([term[i] for i in range(len(term)) if cpt[company.index(c), i] == 1])

    # recommendable_itemsの中で，term1とterm2の両方に含まれる割合を抽出
    term1_index = [i for i in range(len(term)) if term[i] in term1]
    term2_index = [i for i in range(len(term)) if term[i] in term2]
    recommendable_items_index = [i for i in range(len(term)) if term[i] in recommendable_items]
    term1_and_term2 = [i for i in recommendable_items_index if i in term1_index and i in term2_index]
    term1_and_term2_term_list = [term[i] for i in term1_and_term2]
    # cover = len(term1_and_term2) / len(recommendable_items_index)

    # recommendable_itemsの中で，term1に含まれ，term2に含まれない割合を抽出
    term1_not_term2 = [i for i in recommendable_items_index if i in term1_index and i not in term2_index]
    term1_not_term2_term_list = [term[i] for i in term1_not_term2]
    # cover_not_term2 = len(term1_not_term2) / len(recommendable_items_index)

    # recommendable_itemsの中で，term1には含まれず，term2に含まれる割合を抽出
    not_term1_term2 = [i for i in recommendable_items_index if i not in term1_index and i in term2_index]
    not_term1_term2_term_list = [term[i] for i in not_term1_term2]
    # cover_not_term1 = len(not_term1_term2) / len(recommendable_items_index)
    
    # recommendable_itemsの中で，term1には含まれず，term2にも含まれない割合を抽出
    not_term1_not_term2 = [i for i in recommendable_items_index if i not in term1_index and i not in term2_index]
    not_term1_not_term2_term_list = [term[i] for i in not_term1_not_term2]
    # cover_not_term1_not_term2 = len(not_term1_not_term2) / len(recommendable_items_index)

    return term1_and_term2_term_list, term1_not_term2_term_list, not_term1_term2_term_list, not_term1_not_term2_term_list