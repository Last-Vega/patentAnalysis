from networkx.classes.graph import Graph
import numpy as np
import networkx as nx
import pickle
from scipy.sparse import csr_matrix, lil_matrix
from .. import app
temp_folder = app.config['TEMP_FOLDER']
def loadBinary(f_name:str) -> csr_matrix:
    with open(f_name, 'rb') as rb:
        adj:csr_matrix = pickle.load(rb)
    return adj

def createFeatures(adj:csr_matrix) -> lil_matrix:
    n = adj.shape[0]

    features = np.zeros((n, n))
    for i in range(n):
        features[i][i] = 1
    features = lil_matrix(features)
    return features


def forCompare():
    f_name = f'{temp_folder}/adj0306.dict'
    adj_dict = loadBinary(f_name)
    f_name = f'{temp_folder}/bi0306.dict'
    bi_dict = loadBinary(f_name)
    features:lil_matrix = createFeatures(adj_dict['CPC'])
    return adj_dict, bi_dict, features


f_name:str = './W_adj.adj'

# f_name = f'{temp_folder}/W_adj0122.adj'
f_name = f'{temp_folder}/W_adj1213.adj'
adj:csr_matrix = loadBinary(f_name)

features:lil_matrix = createFeatures(adj)

# f_name:str = f'{temp_folder}/W_bi0122.adj'
f_name:str = f'{temp_folder}/W_bi1213.adj'
bi_adj:csr_matrix = loadBinary(f_name)

f_name = f'{temp_folder}/adj0122.dict'
adj_dict = loadBinary(f_name)
f_name = f'{temp_folder}/bi0122-1.dict'
bi_dict = loadBinary(f_name)

# f_name = f'{temp_folder}/adj0304.dict'
# adj_dict = loadBinary(f_name)
# f_name = f'{temp_folder}/bi0304.dict'
# bi_dict = loadBinary(f_name)
# features:lil_matrix = createFeatures(adj_dict['CPC'])

# with open('../vars/annotate.ct', 'rb') as rb:
#     annotate_list:list = pickle.load(rb)
