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

FLAG = True # True: new data, False: old data
if FLAG:
    graph = load_binary(f'{prediction_folder}/new2/bipartite_graph_new.pkl')
    num_company:int = len({n for n, d in graph.nodes(data=True) if d["bipartite"] == 'company'})
    # company:list = list(graph.nodes)[:num_company]
    # term:list = list(graph.nodes)[num_company:]
    company:list = load_binary(f'{prediction_folder}/new2/company_node_new.pkl')
    term:list = load_binary(f'{prediction_folder}/new2/term_node_new.pkl')

    # /home/watanabe/patentAnalysis/backend/4prediction
    # load data
    # patent_company = load_binary(f'{prediction_folder}/new2/patent_company_new.pkl')
    # patent_term = load_binary(f'{prediction_folder}/new2/patent_term_new.pkl')
    cpc = load_binary(f'{prediction_folder}/new2/cpc_new.pkl')
    cpt = load_binary(f'{prediction_folder}/new2/cpt_new.pkl').T
else:
    graph = load_binary(f'{prediction_folder}/bipartite_graph.pkl')
    num_company:int = len({n for n, d in graph.nodes(data=True) if d["bipartite"] == 'company'})
    # company:list = list(graph.nodes)[:num_company]
    # term:list = list(graph.nodes)[num_company:]
    company:list = load_binary(f'{prediction_folder}/company_node.pkl')
    term:list = load_binary(f'{prediction_folder}/term_node.pkl')

    # /home/watanabe/patentAnalysis/backend/4prediction
    # load data
    patent_company = load_binary(f'{prediction_folder}/patent_company.pkl')
    patent_term = load_binary(f'{prediction_folder}/patent_term.pkl')
    cpc = clamp(torch.matmul(patent_company.T, patent_company), 0, 1).to('cpu')
    cpt = clamp(torch.matmul(patent_company.T, patent_term), 0, 1).to('cpu')



def train(model, optimizer, epoch, feature, adj_norm, bi_adj_norm, adj_label, biadj_label, norm, norm_bi, weight_tensor, weight_tensor_bi, collaborated_company_index, target_company_index):
    history = {'loss':[], 'acc':[], 'ap':[], 'roc':[]}

    model.train()
    for i in range(epoch):
        z_c, z_t, A_pred, Bi_pred = model(feature, adj_norm, bi_adj_norm)

        users_zc = z_c.clone()

        for i in collaborated_company_index:
            # print(i, target_company_index)
            users_zc[i] = z_c[target_company_index]
        
        optimizer.zero_grad()

        loss = model.loss_function(norm, norm_bi, adj_label, biadj_label, A_pred, Bi_pred, weight_tensor, weight_tensor_bi, users_zc)

        loss.backward()
        optimizer.step()

        embedding_company = z_c.to('cpu').detach()
        embedding_term = z_t.to('cpu').detach()
    return embedding_company, embedding_term

def prediction(target_company:str, collaborated_company_list:List)->List[str]:
    fix_seed()

    collaborated_company_index:int = [company.index(c) for c in collaborated_company_list]
    
    # collaborated_company_index.append(company.index(target_company))

	# 熊谷組's index
    target_company_index = company.index(target_company)
	
    def print_shape(data_name, data):
        return print(f'{data_name} shape: {data.shape}')

    if FLAG:
        cpc = load_binary(f'{prediction_folder}/new2/cpc_new.pkl')
        cpt = load_binary(f'{prediction_folder}/new2/cpt_new.pkl')

        adj_norm = load_binary(f'{prediction_folder}/new2/adj_norm.pkl')
        adj_label = load_binary(f'{prediction_folder}/new2/adj_label.pkl')
        norm = load_binary(f'{prediction_folder}/new2/norm.pkl')
        weight_tensor = load_binary(f'{prediction_folder}/new2/weight_tensor.pkl')
        bi_adj_norm = load_binary(f'{prediction_folder}/new2/bi_adj_norm.pkl')
        biadj_label = load_binary(f'{prediction_folder}/new2/biadj_label.pkl')
        norm_bi = load_binary(f'{prediction_folder}/new2/norm_bi.pkl')
        weight_tensor_bi = load_binary(f'{prediction_folder}/new2/weight_tensor_bi.pkl')

        feature = create_feature(cpc.shape[1])
        feature = prepare_features_for_training(feature)

        input_dim = feature.shape[1]
        bipartite_dim = cpt.shape[1]

        adj_norm = adj_norm.to(device)
        adj_label = adj_label.to(device)
        biadj_label = biadj_label.to(device)
        bi_adj_norm = bi_adj_norm.to(device)
        feature = feature.to(device)
        
        dropout = 0.0

    else:
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

        adj_norm = adj_norm.to(device)
        adj_label = adj_label.to(device)
        biadj_label = biadj_label.to(device)
        bi_adj_norm = bi_adj_norm
        feature = feature.to(device)
        
        dropout = 0.0
    
    print_shape('adj_norm', adj_norm)
    print_shape('adj_label', adj_label)
    print_shape('biadj_label', biadj_label)
    print_shape('bi_adj_norm', bi_adj_norm)
    print_shape('feature', feature)
    print_shape('cpc', cpc)
    print_shape('cpt', cpt)
    # print_shape('norm', norm)
    # print_shape('norm_bi', norm_bi)
    print_shape('weight_tensor', weight_tensor)
    print_shape('weight_tensor_bi', weight_tensor_bi)
    print(input_dim)
    print(bipartite_dim)
    # print_shape('dropout', dropout)

    model = HeteroVGAE(input_dim, bipartite_dim, hidden1_dim=4, hidden2_dim=2, dropout=dropout)
    model = model.to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    z_c, z_t = train(model, optimizer, 300, feature, adj_norm, bi_adj_norm, adj_label, biadj_label, norm, norm_bi, weight_tensor, weight_tensor_bi, collaborated_company_index, target_company_index)
    print(z_t.shape)
    recommended_items_index = recommendable_items(z_c, z_t, target_company_index, 100)
    recommended_items = [term[i] for i in recommended_items_index]

    z_c = z_c.to('cpu').detach().numpy().copy().tolist()
    z_t = z_t.to('cpu').detach().numpy().copy().tolist()
    
    company_info = []
    
    for i in collaborated_company_index:
        company_info.append({'company':company[i], 'x':z_c[i][0], 'y':z_c[i][1]})
    company_info.append({'company':company[target_company_index], 'x':z_c[target_company_index][0], 'y':z_c[target_company_index][1]})
    term_info = eval(target_company_index, collaborated_company_index, recommended_items, recommended_items_index, z_t, target_company, collaborated_company_list)
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

def eval(target_company_index, collaborated_company_index, recommendable_items, recommended_items_index, z_t, target_company, collaborated_company_list):

	# term1 -> kumagaigumi
    # term1 = [term[i] for i in range(len(term)) if cpt[target_company_index, i] == 1]
    term1 = []
    print(cpt.shape)
    print(len(term))
    for i in range(len(term)):
        if cpt[target_company_index, i] == 1:
            print(i, term[i])
            term1.append(term[i])
    # term2 -> collaborated_company_list
    term2 = []
    for c in collaborated_company_index:
        term2.extend([term[i] for i in range(len(term)) if cpt[c, i] == 1])

    term1_index = [i for i in range(len(term)) if term[i] in term1]
    term2_index = [i for i in range(len(term)) if term[i] in term2]

    term_info = []
    # for i in range(len(recommendable_items)):
    #     term_info.append({'term':term[i], 'color':check_term(term1_index, term2_index, recommended_items_index[i]), 'x':z_t[i][0], 'y':z_t[i][1]})
    for i in recommended_items_index:
        # term_info.append({'term':term[i], 'color':check_term(term1_index, term2_index, i), 'x':z_t[i][0], 'y':z_t[i][1]})
        color_flag = check_term(term1_index, term2_index, i)
        if color_flag == 't1':
            label = f'両方言及'
        elif color_flag == 't2':
            label = f'{target_company}のみ言及'
        elif color_flag == 't3':
            label = f'{(",").join(collaborated_company_list)}のみ言及'
        else:
            label = f'両方言及なし'
        term_info.append({'term':term[i], 'color':color_flag, 'x':z_t[i][0], 'y':z_t[i][1], 'label':label})
            
    return term_info
