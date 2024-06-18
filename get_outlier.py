import itertools
import numpy as np
import random
from datetime import datetime
from collections import Counter


def generate_paired_indices(Y_list, mask, pair_num, num_view):
    num_rows = mask.shape[0]
    half_view = num_view // 2
    all_pairs = list(itertools.combinations(range(num_rows), 2))
    valid_pairs = []
    
    for i, j in all_pairs:
        if (
            np.sum(mask[i] & mask[j]) >= half_view and 
            Y_list[i] != Y_list[j] and
            np.sum(mask[i]) >= 2 and
            np.sum(mask[j]) >= 2
        ):
            valid_pairs.append((i, j))
    
    if len(valid_pairs) < pair_num:
        raise ValueError("Not enough valid pairs available")
    
    selected_pairs = []
    used_indices = set()

    while len(selected_pairs) < pair_num:
        if len(valid_pairs) == 0:
            raise ValueError("Not enough unique pairs available to satisfy pair_num requirement")

        
        idx = np.random.choice(len(valid_pairs))
        pair = valid_pairs[idx]

        
        if pair[0] not in used_indices and pair[1] not in used_indices:
            selected_pairs.append(pair)
            used_indices.update(pair)
        
        
        valid_pairs.pop(idx)

    return selected_pairs



def replace_indices(paired_indices_comp, all_one_indices):
    
    mapping = {old: new for old, new in enumerate(all_one_indices)}
    
    
    replaced_paired_indices_comp = np.vectorize(mapping.get)(paired_indices_comp)
    
    return replaced_paired_indices_comp

def get_outlier(X_list,mask,class_outlier_rate,Y_list):
    
 
    Y_list_np=Y_list[0]

    num_cls_ol=int(class_outlier_rate*mask.shape[0])
    
    pair_num=num_cls_ol//2
    num_cls_ol=pair_num*2
    
    T=np.unique(Y_list_np)
    T=T.shape[0]
    num_view=len(X_list)
    paired_indices=generate_paired_indices(Y_list_np,mask,pair_num,num_view)
    paired_indices=np.array(paired_indices)

    count=0
    for pair in paired_indices:
        
        count=count+1
        
        common_views = np.where(mask[pair[0]] & mask[pair[1]] == 1)[0]
        swap_views = np.random.choice(common_views, num_view//2, replace=False)
        for swap_view in swap_views:
            temp=X_list[swap_view][pair[0]].copy()
            X_list[swap_view][pair[0]]=X_list[swap_view][pair[1]]
            X_list[swap_view][pair[1]]=temp
            

        
    
    out_label = np.zeros(mask.shape[0])
    if paired_indices.size!=0:
        selected_indices=paired_indices.flatten()
        out_label[selected_indices] = 1
        
    
    return X_list,out_label



def get_attribute_outlier_rand(X_list,mask,class_outlabel,attribute_outlier_rate):
    
    num_attr_ol=int(attribute_outlier_rate*mask.shape[0])
    
    class_labels=np.where(class_outlabel==1)[0]



    attr_indices=[item for item in range(mask.shape[0]) if item not in class_labels]
    
    
    selected_indices = np.random.choice(attr_indices, num_attr_ol, replace=False)
    
    for view in X_list:
        rows, cols = view[selected_indices].shape
        view[selected_indices]=np.random.rand(rows, cols)
 
    
    class_outlabel[selected_indices]=1

    out_label=class_outlabel
    
    return X_list,out_label


def get_attr_class_outlier_rand(X_list,mask,attr_class_outlabel,attr_class_outlier_rate,Y_list):
    
    Y_list_np=Y_list[0]
    num_attr_cl_ol=int(attr_class_outlier_rate*mask.shape[0])
    
    attr_class_labels=np.where(attr_class_outlabel==1)[0]
    class_attr_indices=[item for item in range(mask.shape[0]) if item not in attr_class_labels]

    
    
    Y_list_np=Y_list[0]
    pair_num=num_attr_cl_ol//2
    num_attr_cl_ol=pair_num*2
    Y_list_ca=Y_list_np[class_attr_indices]
    mask_ca=mask[class_attr_indices]
    
    num_view=len(X_list)
    paired_indices_ca=generate_paired_indices(Y_list_ca, mask_ca,pair_num,num_view)

    if len(paired_indices_ca)>0:
        paired_indices=replace_indices(paired_indices_ca,class_attr_indices)
    else:
        paired_indices=[]
        paired_indices=np.array(paired_indices)
    

    
    
    for pair in paired_indices:

        common_views = np.where(mask[pair[0]] & mask[pair[1]] == 1)[0]
        swap_views = np.random.choice(common_views, num_view//2, replace=False)
        rest_views=[item for item in range(num_view) if item not in swap_views]
        for swap_view in swap_views:
            temp=X_list[swap_view][pair[0]].copy()
            X_list[swap_view][pair[0]]=X_list[swap_view][pair[1]]
            X_list[swap_view][pair[1]]=temp

        for rest_view in rest_views:
            X_list[rest_view][pair[0]]=np.random.rand(X_list[rest_view].shape[1])
            X_list[rest_view][pair[1]]=np.random.rand(X_list[rest_view].shape[1])


    if paired_indices.size!=0:
        selected_indices=paired_indices.flatten()
        attr_class_outlabel[selected_indices]=1
    
    out_label=attr_class_outlabel
   
    return X_list,out_label



if __name__ == '__main__':
    
    random.seed(1)
    np.random.seed(2)
    x1_train=np.random.rand(5,1)
    x2_train=np.random.rand(5,2)
    X_list=[x1_train,x2_train]
    Y_list=[np.array([1,2,1,2,1])]
    print("previous:\n")
    print(X_list)
    mask=np.array([[1,0],
          [1,1],
          [1,1],
          [1,1],
          [1,1]])
    print(mask.shape)
    rate=0.5
    X_list,out_label=get_outlier(X_list,mask,rate,Y_list)
    print("middle:\n")
    print(X_list)
    X_list,out_label=get_attr_class_outlier_rand(X_list,mask,out_label,0.5,Y_list)
    print("after:\n")
    print(X_list)

