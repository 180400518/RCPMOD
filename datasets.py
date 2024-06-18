import os, random, sys
import numpy as np
import scipy.io as sio
import util
from scipy import sparse


def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
       

    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2]:
            x = train_x[view][index]
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')[index]
        Y_list.append(y)


    elif data_name in ['Caltech101-20','Caltech101']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [0,1,2,3,4,5]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)

    elif data_name in ['Handwritten']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'handwritten.mat'))
        
        X = mat['X']
        for view in [0,1,2,3,4,5]:
            x = X[0][view]
            x = util.normalize(x).astype('float32')
            X_list.append(x)
        
        y = np.squeeze(mat['Y']).astype('int')
        Y_list.append(y)
    
    elif data_name in ['Fashion']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        Y = mat['Y'].astype(np.int32).reshape(10000,)
        V1 = mat['X1'].astype(np.float32)
        V2 = mat['X2'].astype(np.float32)
        V3 = mat['X3'].astype(np.float32)
        V1=V1.reshape(V1.shape[0], 784)
        V2=V2.reshape(V2.shape[0], 784)
        V3=V3.reshape(V3.shape[0], 784)
        X_list.append(V1)
        X_list.append(V2)

        Y_list.append(Y)
    
    elif data_name in ['BDGP']:
        bdgp_filepath="bdgp.npz"
        data = np.load(bdgp_filepath)
        num_views = int(data['n_views'])
        for i in range(num_views):
            x = data[f"view_{i}"]
            if len(x.shape) > 2:
                x = x.reshape([x.shape[0], -1])
            X_list.append(x.astype(np.float32))
        Y_list.append(data['labels']) 
    return X_list, Y_list
