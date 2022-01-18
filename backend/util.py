import numpy as np
from scipy.spatial import distance

def calcCCDistance(latent_c, q_id):
    K = 5
    CCdist = distance.cdist(latent_c, latent_c, metric='euclidean')
    CCdist[q_id][q_id] = 999
    
    close_company_index = np.argpartition(CCdist[q_id], K)[:K-len(CCdist[q_id])].tolist()

    return close_company_index

def calcTTDistance(latent_t, q_id):
    K = 5
    TTdist = distance.cdist(latent_t, latent_t, metric='euclidean')
    TTdist[q_id][q_id] = 999
    
    close_term_index = np.argpartition(TTdist[q_id], K)[:K-len(TTdist[q_id])].tolist()

    return close_term_index

def calcCTDistance(latent_c, latent_t, q_id):
    K = 5
    CTdist = distance.cdist(latent_c, latent_t, metric='euclidean')
    
    close_index = np.argpartition(CTdist[q_id], K)[:K-len(CTdist[q_id])].tolist()

    return close_index

def appendElm(index_list, target_list):
    list = []
    for index in index_list:
        list.append(target_list[index])
    
    return list
