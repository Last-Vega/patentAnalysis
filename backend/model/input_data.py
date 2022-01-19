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

# f_name:str = './W_adj.adj'

f_name = f'{temp_folder}/W_adj.adj'
adj:csr_matrix = loadBinary(f_name)

features:lil_matrix = createFeatures(adj)

f_name:str = f'{temp_folder}/W_bi.adj'
bi_adj:csr_matrix = loadBinary(f_name)

# with open('../vars/annotate.ct', 'rb') as rb:
#     annotate_list:list = pickle.load(rb)
