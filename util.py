import logging
import os
import numpy as np
import math
import torch
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
import numpy as np

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def next_batch(X1, X2, batch_size,FLG_v1,FLG_v2,KNN_v1,KNN_v2,KIND_v1,KIND_v2,POS_v1,POS_v2,NEG_v1,NEG_v2):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        FLG_v1_batch=FLG_v1[start_idx: end_idx, ...]
        FLG_v2_batch=FLG_v2[start_idx: end_idx, ...]
        POS_v1_batch=POS_v1[start_idx: end_idx, ...]
        POS_v2_batch=POS_v2[start_idx: end_idx, ...]
        KNN_v1_batch=KNN_v1[start_idx: end_idx, ...]
        KNN_v2_batch=KNN_v2[start_idx: end_idx, ...]
        KIND_v1_batch=KIND_v1[start_idx: end_idx, ...]
        KIND_v2_batch=KIND_v2[start_idx: end_idx, ...]
        NEG_v1_batch=NEG_v1[start_idx: end_idx, ...]
        NEG_v2_batch=NEG_v2[start_idx: end_idx, ...]

        
        if batch_x1.shape[0]==1:
            continue
        yield (batch_x1, batch_x2, (i + 1),FLG_v1_batch,FLG_v2_batch,KNN_v1_batch,KNN_v2_batch,KIND_v1_batch,KIND_v2_batch,POS_v1_batch,POS_v2_batch,NEG_v1_batch,NEG_v2_batch)



def cal_std(logger, paras,*arg):
    """Return the average and its std"""
    
    logger.info('OUTAUC:'+ str(arg[0]))
    output = """ OUTAUC {:.2f} std {:.2f} """.format(np.mean(arg[0]) * 100,
                                                                                                np.std(arg[0]) * 100,
                                                                                                )
    
    logger.info(output)
    path="./Results/"
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open (path+paras+".txt","w") as file:
        file.write(output+"\n")
    
    
    
    
    

    
    return

def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x