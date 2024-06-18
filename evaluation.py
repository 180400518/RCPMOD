import os
from sklearn import metrics
import numpy as np
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import sys
from munkres import Munkres
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support,accuracy_score
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
import torch
import matplotlib.pyplot as plt




def get_percentile(scores, threshold):
    per = np.percentile(scores, 100 - int(100 * threshold)) 
    return per

def minmaxNormal(mnist):
    min_val = np.min(mnist)
    max_val = np.max(mnist)
    normalized_array = (mnist- min_val) / (max_val - min_val)
    return normalized_array



def cal_final_scores(recon_scores, knn_scores, weight=0.5):
    
    s1 = (recon_scores - np.min(recon_scores)) / (np.max(recon_scores) - np.min(recon_scores))
    s2 = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores))
    total_scores = weight*s1 + s2
    
    

    return total_scores
