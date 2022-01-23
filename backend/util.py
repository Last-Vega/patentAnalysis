import numpy as np
from scipy.spatial import distance
import pickle
import json

from . import app
temp_folder = app.config['TEMP_FOLDER']

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

def loadBinary(f_name):
    with open(f_name, 'rb') as rb:
        data = pickle.load(rb)
    return data

def calcCTDistanceForRecommend(latent_c, latent_t, q_id):
    K = 30
    CTdist = distance.cdist(latent_c, latent_t, metric='euclidean')
    
    close_index = np.argpartition(CTdist[q_id], K)[:K-len(CTdist[q_id])].tolist()

    return close_index

def appendElmForRecommend(index_list, target_list):
    list = []
    for index in index_list:
        list.append(target_list[index])

    kumagai_term = ['上下方向', '回転可能', '地山', '水平方向', '配設', '移動可能', '貫通孔', '床スラブ', '長手方向', '周方向', 'マイクロフォン', '歩行支援装置', '回転中心', 'シールドトンネル', '支持部材', '延長方向', '補強部材', '補強構造', '連結部材', 'クリーンルーム', '所定間隔', '破砕装置', '管設置装置', '破砕装置用電極', '放電破砕', '粒径', '床構造', '接合構造', '音源方向', '施工コスト', '汚染土壌', '軸線方向', '立設', '揺動可能', 'トンネル施工', '推進方向', '掘削機本体', '幅方向', '支持脚', '揺動', '掘削孔', '液体サイクロン', '免震装置', '遮音性能', '汚染物質', '処理装置', '地盤改良', '放電破砕装置', '基礎杭', '接合部分', '地山内', '電極装置', '振動エネルギー', '有害物質', '着脱可能', '圧力伝達媒体', '画像データ', '石膏ボード', 'セグメントリング', '平面形状', '既設トンネル', '所定位置', '鉄筋コンクリート造', 'ベースプレート', '曲げモーメント', '板部材']

    new_list = []
    for term in list:
        if not term in kumagai_term:
            new_list.append(term)
    return new_list