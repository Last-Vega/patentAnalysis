import os
from typing import List
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from .args import *
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
	# want to collaborate with these companies
	target_company_list.append('株式会社熊谷組')
	target_company_index:int = [company.index(c) for c in target_company_list]

	# 熊谷組's index
	target_company = company.index('株式会社熊谷組')
	# prepare for prediction
	# cpc_prediction = create_prediction_tensor(target_company_index, cpc)
	# cpt_prediction = create_prediction_tensor(target_company_index, cpt).T
	cpc_prediction = cpc
	cpt_prediction = cpt.T

	feature = create_feature_for_prediction(cpc.shape[1], target_company_index)
	feature = prepare_features_for_training(feature)
	input_dim = feature.shape[1]
	bipartite_dim = cpt_prediction.shape[1]

	cpc_prediction = cpc_prediction.to(device)
	cpt_prediction = cpt_prediction.to(device)
	feature = feature.to(device)
	dropout = 0.5

	model = HeteroVGAE(input_dim, bipartite_dim, dropout=dropout)
	model.load_state_dict(torch.load(f'{prediction_folder}/model/model0501.pt'))
	model.eval()
	model = model.to(device)
	z_c, z_t, mean, logstd, mu, sigma = model.predict_z(feature, cpc_prediction, cpt_prediction)

	recommended_items = recommendable_items(z_c, z_t, target_company, 100)
	recommended_items = [term[i] for i in recommended_items]

	return recommended_items

# # want to collaborate with these companies
# target_company_list:list = ['株式会社熊谷組', '大成建設株式会社', '鹿島建設株式会社', '清水建設株式会社', '株式会社大林組']
# target_company_index:int = [company.index(c) for c in target_company_list]

# # 熊谷組's index 
# target_company = company.index(target_company_list[0])

# # prepare for prediction
# cpc_prediction = create_prediction_tensor(target_company_index, cpc)
# cpt_prediction = create_prediction_tensor(target_company_index, cpt).T

# feature = create_feature_for_prediction(cpc.shape[1], target_company_index)
# feature = prepare_features_for_training(feature)
# input_dim = feature.shape[1]
# bipartite_dim = cpt_prediction.shape[1]

# cpc_prediction = cpc_prediction.to(device)
# cpt_prediction = cpt_prediction.to(device)
# feature = feature.to(device)
# dropout = 0.5

# print('Start prediction...')

# model = HeteroVGAE(input_dim, bipartite_dim, dropout=dropout)
# model.load_state_dict(torch.load(f'{prediction_folder}/model/model0501.pt'))
# model.eval()
# model = model.to(device)
# z_c, z_t, mean, logstd, mu, sigma = model.predict_z(feature, cpc_prediction, cpt_prediction)

# print(z_c[target_company_index, :])

# recommended_items = recommendable_items(z_c, z_t, target_company, 30)
# print(recommended_items)
# recommended_items = [term[i] for i in recommended_items]
# print(recommended_items)

