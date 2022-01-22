import numpy as np
import pickle

with open('./W_adj.adj', 'rb') as f:
    adj = pickle.load(f).toarray()

print(type(adj))
print(adj)