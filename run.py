

import argparse
import collections
import copy
import itertools
import torch
import pickle

from model import Completer
from get_mask import get_mask
from get_outlier import get_attr_class_outlier_rand, get_attribute_outlier_rand, get_outlier
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config
from scipy.io import savemat
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support,accuracy_score
np.set_printoptions(threshold=np.inf)


dataset = {
    1: "Scene_15",
    2: "LandUse_21",
    3: "BDGP",
    4: "Handwritten",
    5: "Fashion",
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='3', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='5', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')

args = parser.parse_args()
dataset = dataset[args.dataset]



def minmaxNormal(mnist):
    min_val = np.min(mnist)
    max_val = np.max(mnist)
    normalized_array = (mnist- min_val) / (max_val - min_val)
    return normalized_array



def main():
    
    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger = get_logger()

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    if dataset in ['Fashion','BDGP','Handwritten','LandUse_21','Scene_15']:
        X_list_raw, Y_list_raw = load_data(config)
        if dataset in ['LandUse_21',"Scene_15","Fashion",'BDGP']:
            for i in range(len(X_list_raw)):
                X_list_raw[i]=minmaxNormal(X_list_raw[i])
        num_v=len(X_list_raw)


    for mr in range(len(config['training']['missing_rate'])):
        for bs in range(len(config['training']['batch_size'])):
            for e in range(len(config['training']['epoch'])):
                for l in range(len(config['training']['lr'])):
                    for cort in range(len(config['training']['class_outlier_rate'])):
                        for aort in range(len(config['training']['attr_outlier_rate'])):
                            for acort in range(len(config['training']['attr_class_outlier_rate'])):
 
                                if config['training']['class_outlier_rate'][cort]==config['training']['attr_outlier_rate'][aort] or config['training']['class_outlier_rate'][cort]==config['training']['attr_class_outlier_rate'][acort] or config['training']['attr_outlier_rate'][aort]==config['training']['attr_class_outlier_rate'][acort]:
                                    continue
         
                                for nnb in config['training']['num_neibs']:
      
                                    for start_seed in config['training']['start_seed']:
                                        accumulated_metrics = collections.defaultdict(list)
                                        num_test_time=0    
                                        arate=config['training']['attr_outlier_rate'][aort]
                                        crate=config['training']['class_outlier_rate'][cort]
                                        acrate=config['training']['attr_class_outlier_rate'][acort]

                                        paras="mr"+"_"+str(config['training']['missing_rate'][mr])+"_"+"_"+"aort"+"_"+str(arate)+"cort"+"_"+str(crate)+"ancort"+"_"+str(acrate)
                                        
                                        for data_seed in range(start_seed, args.test_time +start_seed):
                                            
                                            np.random.seed(data_seed)
                                            random.seed(data_seed + 1)
                                    
                                            Y_list_seed=copy.deepcopy(Y_list_raw)


                                            X_list=copy.deepcopy(X_list_raw)
                                            

                                            mask = get_mask(num_v, X_list[0].shape[0], config['training']['missing_rate'][mr])
                                            
                                            
                
                                            
                                            class_outlier_rate=config['training']['class_outlier_rate'][cort]
                                            
                                            if  not class_outlier_rate==0:

                                                X_list,class_outlabel=get_outlier(X_list,mask,class_outlier_rate,Y_list_seed)
                                            else:
                                                class_outlabel=np.zeros(mask.shape[0])


        
                                            attr_outlier_rate=config['training']['attr_outlier_rate'][aort]
                                            
                                            
                                            X_list,attr_class_outlabel=get_attribute_outlier_rand(X_list,mask,class_outlabel.copy(),attr_outlier_rate)

                                            for i in range(num_v):
                                                X_list[i]=X_list[i]*mask[:,i][:,np.newaxis]
                                                X_list[i]=minmaxNormal(X_list[i])

                                            attr_class_outlier_rate=config['training']['attr_class_outlier_rate'][acort]
                                            if not attr_class_outlier_rate==0:
                                                X_list,outlabel=get_attr_class_outlier_rand(X_list,mask,attr_class_outlabel.copy(),attr_class_outlier_rate,Y_list_seed)
                                                for i in range(num_v):
                                                    X_list[i]=X_list[i]*mask[:,i][:,np.newaxis]
                                                    X_list[i]=minmaxNormal(X_list[i])
                                            else:
                                                outlabel=attr_class_outlabel.copy()
                                            
                                            
                                            print("total ot:",np.sum(outlabel))
                                            print("class ot:",np.sum(class_outlabel))
                                            print("attr and class ot:",np.sum(attr_class_outlabel))




                                            
                                            COMPLETERCOMP = Completer(config,num_v)
                                            COMPLETERCOMP.to_device(device)
                                            views = list(range(0, num_v))
                                            C_v = list(itertools.combinations(views, 2))
                                            
                                            all_vc_scores=np.zeros((mask.shape[0],len(C_v)))
                                    
                                            for i in range(len(C_v)):
                                                vc=C_v[i]
                                                

                                                optimizerCOMP = torch.optim.Adam(
                                                    itertools.chain(COMPLETERCOMP.autoencoders[vc[0]].parameters(), COMPLETERCOMP.autoencoders[vc[1]].parameters(),),
                                                    lr=config['training']['lr'][l])
                                                x1_train=X_list[vc[0]]
                                                x2_train=X_list[vc[1]]
                                                
                                                mask_vc=mask[:,vc]


                                                x1_train = x1_train * mask_vc[:, 0][:, np.newaxis]

                                                

                                                x2_train = x2_train * mask_vc[:, 1][:, np.newaxis]

                                                
                                                mask_vc_flag=np.sum(mask_vc,axis=1)
                                                mask_vc_flag=np.where(mask_vc_flag!=0)[0]


                                                mask_vc=mask_vc[mask_vc_flag]
                                                outlabel_vc=outlabel[mask_vc_flag]
                                                class_outlabel_vc=class_outlabel[mask_vc_flag]

                                                x1_train=x1_train[mask_vc_flag]
                                                x2_train=x2_train[mask_vc_flag]

                                                                                                                    

                                                x1_train=torch.tensor(x1_train, dtype=torch.float64).to(device)
                                                x2_train=torch.tensor(x2_train, dtype=torch.float64).to(device)


                                                print(x1_train.dtype)

                                                mask_vc = torch.from_numpy(mask_vc).long().to(device)


                                                seed=data_seed
                                                np.random.seed(seed)
                                                random.seed(seed + 1)
                                                torch.manual_seed(seed + 2)
                                                torch.cuda.manual_seed(seed + 3)
                                                torch.cuda.manual_seed_all(seed+4)

                                                torch.backends.cudnn.benchmark = False

                                                torch.backends.cudnn.deterministic = True

                                            
                                                
                                                is_comp=False


                                                

                                                auc_vc,scores_vc= COMPLETERCOMP.train(config, logger, x1_train, x2_train,
                                                                                mask_vc, optimizerCOMP, device,outlabel_vc,
                                                                                e,bs,nnb,num_test_time,paras,seed,Y_list_seed,vc)
                                                
                                                all_vc_scores[mask_vc_flag,i]=scores_vc
                                            
                                            scores = []
                                            for row in all_vc_scores:
                                                non_zero_elements = row[row != 0]
                                                if len(non_zero_elements) > 0:
                                                    mean = np.mean(non_zero_elements)
                                                else:
                                                    mean = 0  
                                                scores.append(mean)
                                            scores=np.array(scores)
                                            
                                            auc=roc_auc_score(outlabel,scores)
                                            print('seed'+str(data_seed)+'AUC:'+str(auc))

                                            accumulated_metrics['outauc'].append(auc)

                                            
                                            
                                                
                                            num_test_time=num_test_time+1 
                                            
                                        
                                        paras_result=paras+config['training']['dataset_name']+"MN"+str(config['training']['Moreratio'])+"K"+str(config['training']['num_neibs'])+"LOSSPARA"
                                        for para in config['training']['loss_paras']:
                                            paras_result+=str(para)                                   
                                                    
                                                    
                                        logger.info('--------------------Training over--------------------')
                                    
                                        paras_result=paras+config['training']['dataset_name']+"MN"+str(config['training']['Moreratio'])+"MS"+str(config['training']['Memorysize'])+"K"+str(config['training']['num_neibs'])+"LOSSPARA"
                                        for para in config['training']['loss_paras']:
                                            paras_result+=str(para)

                                                        
                                        cal_std(logger,paras_result,accumulated_metrics['outauc'])


if __name__ == '__main__':
    main()
