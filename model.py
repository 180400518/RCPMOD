
import math
import pickle
import sys
import time
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import  contrastive_loss_oa,contrastive_loss,contrastive_score,triplet_loss, uniform_loss, uniform_loss_save
import evaluation
from util import next_batch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.neighbors import kneighbors_graph
import copy
np.set_printoptions(threshold=np.inf)
import os
from evaluation import cal_final_scores,get_percentile
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support,accuracy_score,normalized_mutual_info_score
from sklearn.neighbors import LocalOutlierFactor



class Autoencoder(nn.Module):

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim-1):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Linear(decoder_dim[len(decoder_dim)-2], decoder_dim[len(decoder_dim)-1]))
        decoder_layers.append(nn.Sigmoid())

        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent





class RCPMOD():

    def __init__(self,
                 config,num_views):
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        
        
        self.autoencoders=[]
        for i in range(num_views):
            arch='arch'+str(i+1)
            self.autoencoders.append(Autoencoder(config['Autoencoder'][arch], config['Autoencoder']['activations'],
                                        config['Autoencoder']['batchnorm']))

        

        self.softmax = nn.Softmax(dim=1)
        
        for i in range(num_views): 
            self.autoencoders[i].to(torch.float64)
        


    def to_device(self, device):
        """ to cuda if gpu is used """
        for i in range(len(self.autoencoders)): 
            self.autoencoders[i].to(device)



    def minmaxNormal(self,mnist):
        min_val = np.min(mnist)
        max_val = np.max(mnist)
        normalized_array = (mnist- min_val) / (max_val - min_val)
        return normalized_array


    def get_knn_indices(self,flag_v1,flag_v2,x1_train,x2_train,nnb,neigh_number_large=140):
        exist_v1=np.where(flag_v1.cpu().numpy()==1)[0]
        exist_v2=np.where(flag_v2.cpu().numpy()==1)[0]
        ms_v1=np.where(flag_v1.cpu().numpy()==0)[0]
        ms_v2=np.where(flag_v2.cpu().numpy()==0)[0]
        
        x1_train_exist=x1_train[exist_v1]
        x2_train_exist=x2_train[exist_v2]

        
        if nnb<10:
            neigh_number=40
        else:
            neigh_number=neigh_number_large
        
        
        neigh1 = NearestNeighbors(n_neighbors=neigh_number)
        neigh2 = NearestNeighbors(n_neighbors=neigh_number)
        x1_train_existnp=x1_train_exist.cpu().numpy()
        x2_train_existnp=x2_train_exist.cpu().numpy()
        neigh1.fit(x1_train_existnp)
        neigh2.fit(x2_train_existnp)
        _, indicesv1_exist = neigh1.kneighbors(x1_train_existnp)
        _, indicesv2_exist = neigh2.kneighbors(x2_train_existnp)
        indicesv1_exist = indicesv1_exist[:, 1:]
        indicesv2_exist = indicesv2_exist[:, 1:]
        
        indicesv1=np.full((indicesv1_exist.shape[0],indicesv1_exist.shape[1]), -1, dtype=int)
        indicesv2=np.full((indicesv2_exist.shape[0],indicesv2_exist.shape[1]), -1, dtype=int)


        
        for i in range(indicesv1_exist.shape[0]):
            for j in range(indicesv1_exist.shape[1]): 
                indicesv1[i, j] = exist_v1[indicesv1_exist[i, j]]
        
        for i in range(indicesv2_exist.shape[0]):
            for j in range(indicesv2_exist.shape[1]):
                indicesv2[i, j] = exist_v2[indicesv2_exist[i, j]]


        num_k=nnb
        full_indicesv1 = np.full((x1_train.shape[0],indicesv1.shape[1]), -1, dtype=int)
        full_indicesv2 = np.full((x2_train.shape[0],indicesv2.shape[1]), -1, dtype=int)
        k_indicesv1=np.full((x1_train.shape[0],num_k), -1, dtype=int)
        k_indicesv2=np.full((x2_train.shape[0],num_k), -1, dtype=int)
        
        
        for idx, original_idx in enumerate(exist_v1):
            full_indicesv1[original_idx] = indicesv1[idx]

        for idx, original_idx in enumerate(exist_v2):
            full_indicesv2[original_idx] = indicesv2[idx]


        

        values_to_fillv1 = full_indicesv2[ms_v1]
        new_values_to_fillv1 = []
        new_values_to_fillv2 = []

        for row in values_to_fillv1:
            
            filtered_row = np.array([val for val in row if val not in ms_v1])
            
            
            if len(filtered_row) > num_k:
                filtered_row = filtered_row[:num_k]
            
            new_values_to_fillv1.append(filtered_row)

        for idx, value in zip(ms_v1, new_values_to_fillv1):
            k_indicesv1[idx] = value

        values_to_fillv2 = full_indicesv1[ms_v2]
        for row in values_to_fillv2:
            
            filtered_row = np.array([val for val in row if val not in ms_v2])
            
            
            if len(filtered_row) > num_k:
                filtered_row = filtered_row[:num_k]
            
            new_values_to_fillv2.append(filtered_row)

        for idx, value in zip(ms_v2, new_values_to_fillv2):
            k_indicesv2[idx] = value

        

        k_indicesv1[exist_v1]=full_indicesv1[exist_v1, :num_k]
        k_indicesv2[exist_v2]=full_indicesv2[exist_v2, :num_k]


       
        return k_indicesv1,k_indicesv2

    def linear_scheduler(self,base_values, final_values, total_iter, warmup_iter=0, start_warmup_value=0, schedule_global=True):
        if not schedule_global:
            final_values[0] = base_values[0]
        res = []
        for base_value, final_value in zip(base_values, final_values):
            iters = total_iter - warmup_iter
            schedule = np.linspace(base_value, final_value, iters)


            if warmup_iter > 0:
                warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iter)
                schedule = np.concatenate((warmup_schedule, schedule))
            

            assert len(schedule) == total_iter
            res.append(schedule)

        return np.array(res).T


    def train(self, config, logger, x1_train, x2_train,  mask, optimizer, device,outlabel,
              e,bs,nnb,num_test_time,paras,dataseed,Y_list,vc):

        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        flag_v1=flag[:,0]
        flag_v2=flag[:,1]
        k_indicesv1,k_indicesv2=self.get_knn_indices(flag_v1,flag_v2,x1_train,x2_train,nnb)
        x_knn_v1=x1_train[k_indicesv1]
        x_knn_v2=x2_train[k_indicesv2]
        k_indicesv1_ori=k_indicesv1
        k_indicesv2_ori=k_indicesv2

        if config['training']['dataset_name']=='SCENE15':
            neigh_number_large=130
        else:
            neigh_number_large=110
        neg_indicesv1,neg_indicesv2=self.get_knn_indices(flag_v1,flag_v2,x1_train,x2_train,50,neigh_number_large)
        neg_idx_v1=neg_indicesv1[:,-1]
        neg_idx_v2=neg_indicesv2[:,-1]
        neg_v1=x1_train[neg_idx_v1]
        neg_v2=x2_train[neg_idx_v2]

        
        memory_v1=[]
        memory_v2=[]

    
        now = datetime.now()
        today = datetime.today()
        formatted_date = today.strftime("%Y-%m-%d")
        formatted_time = now.strftime("%H-%M-%S")

        
        datanum=x1_train.shape[0]
        classnum=np.unique(Y_list).shape[0]


        logname='../newlogs/'+formatted_date+"-"+formatted_time+str(num_test_time)+"SEED"+str(dataseed)+paras+config['training']['Run_name']+config['training']['dataset_name']+str(datanum)+"C"+str(classnum)+"vc"+str(vc)
        
        plusname=""
        if config['training']['Using_Contr']:
            plusname+="Contr"
        if config['training']['Using_K_Contr']:
            plusname+="K_Contr"
        if config['training']['Using_KeLeo']:
            plusname+="KL"
        if config['training']['Using_KeLeo']:
            plusname+="Tri"
       
        plusname+=str(config['training']['Memorysize'])+str(config['training']['Moreratio'])
        if config['training']['Using_schedule']:
            plusname+="WM"
            plusname+=str(config['training']['warmup_max'])
            plusname+='Scheepoch'
            plusname+=str(config['training']['schedule_epoch'])

        plusname+="Para"
        for para in config['training']['loss_paras']:
            plusname+=str(para)

        logname=logname+plusname
        tsnename=config['training']['dataset_name']+'/'+formatted_date+"-"+formatted_time+config['training']['Run_name']+paras+plusname


        
        print(logname)
        
        writer = SummaryWriter(logname) 

        n_iter=0
        
        

         
        training_epochs=config['training']['epoch'][e]


        if config['training']['Using_schedule']:
            Schedule_epoch=config['training']['schedule_epoch']
            KeLeo_schedule = self.linear_scheduler([config['training']['warmup_max']], [config['training']['loss_paras'][2]],  Schedule_epoch, warmup_iter=100, start_warmup_value=0, schedule_global=True)

            KeLeo_schedule =torch.tensor(KeLeo_schedule, device=device, dtype=x1_train.dtype)

        
        start_time_allepoch = time.time()
        for epoch in range(training_epochs):
            REP_1 = self.autoencoders[vc[0]].encoder(x1_train).detach()
            REP_2 = self.autoencoders[vc[1]].encoder(x2_train).detach()
            if config['training']['Change_Knn'] and (epoch)%config['training']['change_knn_epoch']==0:
                start_time_getknn=time.time()
                k_indicesv1,k_indicesv2=self.get_knn_indices(flag_v1,flag_v2,REP_1,REP_2,nnb)
                end_time_getknn=time.time()
                if epoch==config['training']['change_knn_epoch']:
                    time_getknn=end_time_getknn-start_time_getknn
                    timepath='../Timetest/'+tsnename+'/'
                    if not os.path.exists(timepath):
                        os.makedirs(timepath)
                    timepath=timepath+'seed'+str(dataseed)+'time.txt'
                    with open(timepath, 'a') as f:
                        f.write(" Getknn:"+str(time_getknn))
                x_knn_v1=x1_train[k_indicesv1]
                x_knn_v2=x2_train[k_indicesv2]

               

            if config['training']['Using_KeLeo'] and (epoch)%config['training']['change_neg_epoch']==0:
                neg_indicesv1,neg_indicesv2=self.get_knn_indices(flag_v1,flag_v2,REP_1,REP_2,50)
                neg_idx_v1=neg_indicesv1[:,-1]
                neg_idx_v2=neg_indicesv2[:,-1]
                neg_v1=x1_train[neg_idx_v1]
                neg_v2=x2_train[neg_idx_v2]
            entropy_v1=None
            entropy_v2=None
            

                
            
            rank_pos = np.random.choice(nnb, size=x1_train.shape[0])
            positive_idx_v1 = k_indicesv1_ori[np.arange(x1_train.shape[0]), rank_pos]
            positive_idx_v2 = k_indicesv2_ori[np.arange(x1_train.shape[0]), rank_pos]
            positive_v1=x1_train[positive_idx_v1]
            positive_v2=x2_train[positive_idx_v2]

            
            
            X1, X2,FLG_v1,FLG_v2,KNN_v1,KNN_v2,KIND_v1,KIND_v2,POS_v1,POS_v2,NEG_v1,NEG_v2 = shuffle(x1_train, x2_train,flag_v1,flag_v2,x_knn_v1,x_knn_v2,k_indicesv1,k_indicesv2,positive_v1,positive_v2,neg_v1,neg_v2,random_state=1)
        
            
            loss_all, loss_rec1, loss_rec2, loss_cl, loss_pre,loss_Contr,loss_trip1,loss_trip2,loss_uniform1,loss_uniform2 = 0, 0, 0, 0, 0,0,0,0,0,0
            
            Z_1fullbatch=self.autoencoders[vc[0]].encoder(X1).detach()
            Z_2fullbatch=self.autoencoders[vc[1]].encoder(X2).detach()
            
            shuffle_flag=True
           
            
            start_time_oneepoch = time.time()
            for batch_x1, batch_x2,batch_No,FLG_v1_batch,FLG_v2_batch,KNN_v1_batch,KNN_v2_batch,KIND_v1_batch,KIND_v2_batch,POS_v1_batch,POS_v2_batch,NEG_v1_batch,NEG_v2_batch in next_batch(X1, X2, config['training']['batch_size'][bs],FLG_v1,FLG_v2,KNN_v1,KNN_v2,KIND_v1,KIND_v2,POS_v1,POS_v2,NEG_v1,NEG_v2):
                z_1 = self.autoencoders[vc[0]].encoder(batch_x1)
                z_2 = self.autoencoders[vc[1]].encoder(batch_x2)

                
                FLG_comp = (FLG_v1_batch + FLG_v2_batch) == 2
                FLG_comp_ind=torch.where(FLG_comp)[0]
                z_1_comp=z_1[FLG_comp]
                z_2_comp=z_2[FLG_comp]

                contr_weights=torch.ones(batch_x1.shape[0]).to(device)


                if config['training']['Using_Contr']:
                    if config['training']['Contr_on_Impu'] and (epoch+1)>config['training']['Contr_on_impu_Stepoch']:
                        z_1ms_idx = FLG_v1_batch == 0
                        z_2ms_idx = FLG_v2_batch == 0
                        rep_1_full = self.cross_transfer_completion(self.autoencoders[vc[0]],z_1,z_2,KNN_v1_batch,z_1ms_idx,config,KIND_v1_batch)
                        rep_2_full = self.cross_transfer_completion(self.autoencoders[vc[1]],z_2,z_1,KNN_v2_batch,z_2ms_idx,config,KIND_v2_batch)               
                        rep_1notnorm=rep_1_full.clone()
                        rep_2notnorm=rep_2_full.clone()
        
                        rep_1_full=F.normalize(rep_1_full, dim=1)
                        rep_2_full=F.normalize(rep_2_full, dim=1)

                        
                        
                            

                        if not len(memory_v1)==0 and not len(memory_v2)==0:
                            memory_v1_cat= torch.cat(memory_v1, dim=0)
                            memory_v2_cat= torch.cat(memory_v2, dim=0)
                            memoryh_v1=self.autoencoders[vc[0]].encoder(memory_v1_cat)
                            memoryh_v2=self.autoencoders[vc[1]].encoder(memory_v2_cat)
                            rep_1_fullnorm,memoryh_v1=norm_and_cut(rep_1notnorm,memoryh_v1)
                            rep_2_fullnorm,memoryh_v2=norm_and_cut(rep_2notnorm,memoryh_v2)
                            
                            contrast_loss=contrastive_loss_oa(rep_1_fullnorm,rep_2_fullnorm,memoryh_v1,memoryh_v2,rep_1_full.shape[0],contr_weights)
                        else:
                            contrast_loss=contrastive_loss(rep_1_full,rep_2_full,rep_1_full.shape[0],contr_weights)
                        
                    
                        
                            
                        repsims=(rep_1_full*rep_2_full).sum(dim=1).squeeze()
                        topnum= max(int(repsims.numel() * config['training']['Moreratio']), 1)
                        _, top10percent = torch.topk(-repsims, topnum)
                        
                        top10percent_v1=batch_x1[top10percent]
                        top10percent_v2=batch_x2[top10percent]
                        
                        memory_v1.append(top10percent_v1)
                        memory_v2.append(top10percent_v2)
                        
                        if len(memory_v1)>config['training']['Memorysize']:
                            memory_v1.pop(0)
                            memory_v2.pop(0)

                    else:    
                        
                        rep_1comp=z_1_comp
                        rep_2comp=z_2_comp
                        rep_1norm=F.normalize(rep_1comp, dim=1)
                        rep_2norm=F.normalize(rep_2comp, dim=1)
                        
                    
                            
                        if not len(memory_v1)==0 and not len(memory_v2)==0:
                            memory_v1_cat= torch.cat(memory_v1, dim=0)
                            memory_v2_cat= torch.cat(memory_v2, dim=0)
                            memoryh_v1=self.autoencoders[vc[0]].encoder(memory_v1_cat)
                            memoryh_v2=self.autoencoders[vc[1]].encoder(memory_v2_cat)
                            rep_1compnorm,memoryh_v1=norm_and_cut(rep_1comp,memoryh_v1)
                            rep_2compnorm,memoryh_v2=norm_and_cut(rep_2comp,memoryh_v2)
                            contrast_loss=contrastive_loss_oa(rep_1compnorm,rep_2compnorm,memoryh_v1,memoryh_v2,rep_1norm.shape[0],contr_weights[FLG_comp])

                        else:
                            contrast_loss=contrastive_loss(rep_1norm,rep_2norm,rep_1norm.shape[0],contr_weights[FLG_comp])    
                          
                        repsims=(rep_1norm*rep_2norm).sum(dim=1).squeeze()
                        topnum= max(int(repsims.numel() * config['training']['Moreratio']), 1)
                        _, top10percent= torch.topk(-repsims, topnum)
                        
                        top10percent=FLG_comp_ind[top10percent]
                        
                        top10percent_v1=batch_x1[top10percent]
                        top10percent_v2=batch_x2[top10percent]
                        
                        memory_v1.append(top10percent_v1)
                        memory_v2.append(top10percent_v2)
                        
                        if len(memory_v1)>config['training']['Memorysize']:
                            memory_v1.pop(0)
                            memory_v2.pop(0)
                                

                
                FLG_v1_exist=(FLG_v1_batch==1)
                FLG_v2_exist=(FLG_v2_batch==1)

                z_1exist=z_1[FLG_v1_exist]
                z_2exist=z_2[FLG_v2_exist]    

                recon1 = F.mse_loss(self.autoencoders[vc[0]].decoder(z_1exist), batch_x1[FLG_v1_exist],reduction="none")
                recon2 = F.mse_loss(self.autoencoders[vc[1]].decoder(z_2exist), batch_x2[FLG_v2_exist],reduction="none")
                
                recon1=torch.sum(recon1,dim=tuple(range(1,batch_x1.dim())))
                recon2=torch.sum(recon2,dim=tuple(range(1,batch_x2.dim())))
                

                recon1=torch.mean(recon1)
                recon2=torch.mean(recon2)
                
                reconstruction_loss = recon1 + recon2

                
                
                if config['training']['Using_K_Contr']:
                    k_infoNCE_loss=0
                    for nk in range(nnb):
                        Kth_neib_v1=KNN_v1_batch[:,nk,:]
                        Kth_neib_v2=KNN_v2_batch[:,nk,:]
                        repk_v1=self.autoencoders[vc[0]].encoder(Kth_neib_v1)
                        repk_v2=self.autoencoders[vc[1]].encoder(Kth_neib_v2)
                        repk_v1=F.normalize(repk_v1,dim=1)
                        repk_v2=F.normalize(repk_v2,dim=1)
                        

                        contr_weights_kinfo=torch.ones(repk_v1.shape[0]).to(device)

                        k_infoNCE_loss+=contrastive_loss(repk_v1,repk_v2,batch_x1.shape[0],contr_weights_kinfo)
                    k_infoNCE_loss=k_infoNCE_loss/nnb



                rep1_exist=z_1exist
                rep2_exist=z_2exist


                if config['training']['Using_KeLeo']:
                    pos_v1_exist=POS_v1_batch[FLG_v1_exist]
                    pos_v2_exist=POS_v2_batch[FLG_v2_exist]
                    neg_v1_exist=NEG_v1_batch[FLG_v1_exist]
                    neg_v2_exist=NEG_v2_batch[FLG_v2_exist]
                    
                    pos_v1_exist=self.autoencoders[vc[0]].encoder(pos_v1_exist)
                    pos_v2_exist=self.autoencoders[vc[1]].encoder(pos_v2_exist)
                    neg_v1_exist=self.autoencoders[vc[0]].encoder(neg_v1_exist)
                    neg_v2_exist=self.autoencoders[vc[1]].encoder(neg_v2_exist)
                    triplet_loss_v1=triplet_loss(rep1_exist,pos_v1_exist,neg_v1_exist)
                    triplet_loss_v2=triplet_loss(rep2_exist,pos_v2_exist,neg_v2_exist)
                    
                    trip_loss=triplet_loss_v1+triplet_loss_v2
                    
                if config['training']['Using_KeLeo'] and (epoch+1)>config['training']['start_Keleo_loss_epoch']:
                    uniform_loss1=uniform_loss(rep1_exist)
                    uniform_loss2=uniform_loss(rep2_exist)
                    uni_loss=uniform_loss1+uniform_loss2
    

                loss =reconstruction_loss
                

                if config['training']['Using_Contr']:
                    loss+=contrast_loss*config['training']['loss_paras'][0]

                if config['training']['Using_K_Contr']:
                    loss+=k_infoNCE_loss*config['training']['loss_paras'][1]
                if config['training']['Using_KeLeo']:
                    if config['training']['Using_schedule']:
                        Keleoweight=KeLeo_schedule[epoch].item()
                        loss+=trip_loss*Keleoweight
                    else:
                        loss+=trip_loss*config['training']['loss_paras'][2]
                if config['training']['Using_KeLeo'] and (epoch+1)>config['training']['start_Keleo_loss_epoch']:
                    
                    if config['training']['Using_schedule']:
                        Keleoweight=KeLeo_schedule[epoch].item()
                        loss+=uni_loss*Keleoweight
                    else:
                        loss+=uni_loss*config['training']['loss_paras'][2]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                if config['training']['Using_KeLeo']:
                    loss_trip1+=triplet_loss_v1.item()
                    loss_trip2+=triplet_loss_v2.item()
                if config['training']['Using_KeLeo'] or config['training']['Save_Keleo_log']:
                    loss_uniform1+=uniform_loss1.item()
                    loss_uniform2+=uniform_loss2.item()

                if config['training']['Using_Contr']:
                    loss_Contr+=contrast_loss.item()
                if config['training']['Using_K_Contr']:
                    loss_Kinfo=k_infoNCE_loss.item()
            end_time_oneepoch = time.time()
            if epoch==0:
                time_oneepoch=end_time_oneepoch-start_time_oneepoch
                timepath='../Timetest/'+tsnename+'/'
                if not os.path.exists(timepath):
                    os.makedirs(timepath)
                timepath=timepath+'seed'+str(dataseed)+'time.txt'
                with open(timepath, 'a') as f:
                    f.write(" Oneepoch:"+str(time_oneepoch))
            
            
            writer.add_scalar('training loss', loss_all, global_step=epoch)
            writer.add_scalar('recon loss1', loss_rec1, global_step=epoch)
            writer.add_scalar('recon loss2', loss_rec2, global_step=epoch)
            writer.add_scalar('pre loss', loss_pre, global_step=epoch)
            writer.add_scalar('cl loss', loss_cl, global_step=epoch)
            if config['training']['Using_Contr']:
                writer.add_scalar('contrast loss',loss_Contr, global_step=epoch)
            if config['training']['Using_K_Contr']:
                writer.add_scalar('K infoNCE loss', loss_Kinfo, global_step=epoch)
            if config['training']['Using_KeLeo']:
                writer.add_scalar('Triplet loss v1', loss_trip1, global_step=epoch)
                writer.add_scalar('Triplet loss v2', loss_trip2, global_step=epoch)
            if config['training']['Using_KeLeo'] or config['training']['Save_Keleo_log']:
                writer.add_scalar('KeLeo loss v1', loss_uniform1, global_step=epoch)
                writer.add_scalar('KeLeo loss v2', loss_uniform2, global_step=epoch)


            
            if (epoch + 1) % config['print_num'] == 0 :

                auc,score_contrrecon=\
                self.evaluation(config, logger, mask, x1_train, x2_train, 
                device,outlabel,bs,epoch,dataseed,k_indicesv1,k_indicesv2,x_knn_v1,x_knn_v2,Y_list,tsnename,vc)
                
                
                
                writer.add_scalar('auc', auc, global_step=epoch)

        end_time_allepoch = time.time()

        time_allepoch=end_time_allepoch-start_time_allepoch

        
        timepath='../Timetest/'+tsnename+'/'
        if not os.path.exists(timepath):
            os.makedirs(timepath)
        timepath=timepath+'seed'+str(dataseed)+'time.txt'
        with open(timepath, 'a') as f:
            f.write(" Allepoch:"+str(time_allepoch))

        return auc,score_contrrecon

    
    

    def evaluation(self, config, logger, mask, x1_train, x2_train, device,outlabel,bs,epoch,dataseed,
                   k_indicesv1,k_indicesv2,x_knn_v1,x_knn_v2,Y_list,tsnename,vc):
        with torch.no_grad():
            
            self.autoencoders[vc[0]].eval(), self.autoencoders[vc[1]].eval()

            img_idx_eval = mask[:, 0] == 1
            txt_idx_eval = mask[:, 1] == 1
            img_missing_idx_eval = mask[:, 0] == 0
            txt_missing_idx_eval = mask[:, 1] == 0
            
        
            all_one_rows = torch.all(mask == 1, axis=1)
            all_one_indices = torch.nonzero(all_one_rows).squeeze()

            all_one_indicesnp=all_one_indices.cpu().numpy()


            
            z_1 = self.autoencoders[vc[0]].encoder(x1_train)
            z_2 = self.autoencoders[vc[1]].encoder(x2_train)        

                
            start_time_completion=time.time()
            rep_1 = self.cross_transfer_completion(self.autoencoders[vc[0]], z_1,z_2, x_knn_v1, img_missing_idx_eval,config,k_indicesv1)
            rep_2 = self.cross_transfer_completion(self.autoencoders[vc[1]], z_2,z_1, x_knn_v2, txt_missing_idx_eval,config,k_indicesv2)               
            end_time_completion=time.time()
        
            time_completion=end_time_completion-start_time_completion
            timepath='../Timetest/'+tsnename+'/'
            if not os.path.exists(timepath):
                os.makedirs(timepath)
            timepath=timepath+'seed'+str(dataseed)+'time.txt'
            with open(timepath, 'a') as f:
                f.write(" Completion:"+str(time_completion))
        

            start_time_Attr=time.time()

            attr_scores=self.batch_calculate_attr_scores(x1_train,x2_train,all_one_indices,config,img_missing_idx_eval,txt_missing_idx_eval,bs,vc,device)
            
            end_time_Attr=time.time()
            time_Attr=end_time_Attr-start_time_Attr
            timepath='../Timetest/'+tsnename+'/'
            if not os.path.exists(timepath):
                os.makedirs(timepath)
            timepath=timepath+'seed'+str(dataseed)+'time.txt'
            with open(timepath, 'a') as f:
                f.write(" Attr:"+str(time_Attr))


            start_time_Contr=time.time()
            Contr_loss_score=contrastive_score(F.normalize(rep_1,dim=1),F.normalize(rep_2,dim=1),rep_1.shape[0])
            Contr_loss_scorenp=Contr_loss_score.cpu().numpy()
            end_time_Contr=time.time()
            time_Contr=end_time_Contr-start_time_Contr
            timepath='../Timetest/'+tsnename+'/'
            if not os.path.exists(timepath):
                os.makedirs(timepath)
            timepath=timepath+'seed'+str(dataseed)+'time.txt'
            with open(timepath, 'a') as f:
                f.write(" Contr:"+str(time_Contr))
           

            start_time_Auc=time.time()
            
            score_contrrecon=cal_final_scores(attr_scores,Contr_loss_scorenp,config['training']['Rw'])
            auc=roc_auc_score(outlabel,score_contrrecon)
            end_time_Auc=time.time()
            time_Auc=end_time_Auc-start_time_Auc
            timepath='../Timetest/'+tsnename+'/'
            if not os.path.exists(timepath):
                os.makedirs(timepath)
            timepath=timepath+'seed'+str(dataseed)+'time.txt'
            with open(timepath, 'a') as f:
                f.write(" Auc:"+str(time_Auc))
                        

            logger.info("\033[2;29m" + 'outauc' + str(auc) + "\033[0m")

            self.autoencoders[vc[0]].train(), self.autoencoders[vc[1]].train()

        return auc,score_contrrecon

    
    def cross_transfer_completion(self,autoencoder, rep_1,rep_2, x_knn, missing_idx_eval,config,k_indices):
        z_knn = autoencoder.encoder(x_knn.view(-1, x_knn.shape[2]))
        rep_knn = z_knn.view(x_knn.shape[0], x_knn.shape[1], z_knn.shape[1]) 
            
        rep_knn_re = rep_knn[missing_idx_eval.cpu().numpy()]
        
        rep_re = rep_knn_re.mean(axis=1)
        rep_1[missing_idx_eval.cpu().numpy()] = rep_re
        return rep_1

    def batch_calculate_attr_scores(self,x1_train,x2_train,all_one_indices,config,img_missing_idx_eval,txt_missing_idx_eval,bs,vc,device,is_weight=False,reverse_weight=False):
        
        complete_v1=x1_train[all_one_indices]
        complete_v2=x2_train[all_one_indices]
        tot = complete_v1.shape[0]
        batch_size=config['training']['batch_size'][bs]
        total = math.ceil(tot /batch_size)
        
        complete_imgs_latent_eval=torch.empty(0,config['Autoencoder']['latent_dim'])
        complete_txts_latent_eval=torch.empty(0,config['Autoencoder']['latent_dim'])
        complete_imgs_recon=torch.empty(0,x1_train.shape[1])
        complete_txts_recon=torch.empty(0,x2_train.shape[1])
        v1_dual_pred=torch.empty(0,config['Autoencoder']['latent_dim'])
        v2_dual_pred=torch.empty(0,config['Autoencoder']['latent_dim'])
        
        for i in range(int(total)):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            end_idx = min(tot, end_idx)
            batch_x1 = complete_v1[start_idx: end_idx, ...]
            batch_x2 = complete_v2[start_idx: end_idx, ...]
            
            complete_imgs_latent_eval_batch=self.autoencoders[vc[0]].encoder(batch_x1)
            complete_txts_latent_eval_batch=self.autoencoders[vc[1]].encoder(batch_x2)
            complete_imgs_recon_batch=self.autoencoders[vc[0]].decoder(complete_imgs_latent_eval_batch)
            complete_txts_recon_batch=self.autoencoders[vc[1]].decoder(complete_txts_latent_eval_batch)
    

            if i == 0:
                complete_imgs_latent_eval = complete_imgs_latent_eval_batch
                complete_txts_latent_eval = complete_txts_latent_eval_batch
                complete_imgs_recon=complete_imgs_recon_batch
                complete_txts_recon=complete_txts_recon_batch
            
                
            else:

                complete_imgs_latent_eval = torch.cat((complete_imgs_latent_eval, complete_imgs_latent_eval_batch), dim=0)
                complete_txts_latent_eval = torch.cat((complete_txts_latent_eval, complete_txts_latent_eval_batch), dim=0)
                complete_imgs_recon = torch.cat((complete_imgs_recon, complete_imgs_recon_batch), dim=0)
                complete_txts_recon = torch.cat((complete_txts_recon, complete_txts_recon_batch), dim=0)
                

        v2_v1missing_latent_eval,v2_v1missing_recon=self.batch_process_autoencoder(x2_train[img_missing_idx_eval],batch_size,self.autoencoders[vc[1]],config['Autoencoder']['latent_dim'],x2_train.shape[1],device)    
        v1_v2missing_latent_eval,v1_v2missing_recon=self.batch_process_autoencoder(x1_train[txt_missing_idx_eval],batch_size,self.autoencoders[vc[0]],config['Autoencoder']['latent_dim'],x1_train.shape[1],device)    
        
        
        
        v2_v1missing_recon_error=torch.sum(( v2_v1missing_recon-x2_train[img_missing_idx_eval])**2, dim=tuple(range(1, v2_v1missing_recon.dim())))
        v1_v2missing_recon_error=torch.sum(( v1_v2missing_recon-x1_train[txt_missing_idx_eval])**2, dim=tuple(range(1, v1_v2missing_recon.dim())))      
                

        complete_imgs_recon_score=torch.sum((complete_imgs_recon - complete_v1)**2, dim=tuple(range(1, complete_v1.dim())))
        complete_txts_recon_score=torch.sum((complete_txts_recon - complete_v2)**2, dim=tuple(range(1, complete_v1.dim())))

        concatenated_v2 = np.concatenate((complete_txts_recon_score.cpu().numpy(), v2_v1missing_recon_error.cpu().numpy()))
        concatenated_v1 = np.concatenate((complete_imgs_recon_score.cpu().numpy(), v1_v2missing_recon_error.cpu().numpy()))
                
        normalized_v2=self.minmaxNormal(concatenated_v2)
        normalized_v1=self.minmaxNormal(concatenated_v1)

        normalized_complete_txts_recon_score = normalized_v2[:len(complete_txts_recon_score)]
        normalized_v2_v1missing_recon_error = normalized_v2[len(complete_txts_recon_score):]
        
        normalized_complete_imgs_recon_score = normalized_v1[:len(complete_imgs_recon_score)]
        normalized_v1_v2missing_recon_error = normalized_v1[len(complete_imgs_recon_score):]

        v1_missing_samev2_reconerror=normalized_v2_v1missing_recon_error
        v2_missing_samev1_reconerror=normalized_v1_v2missing_recon_error  


        imgs_recon_score=np.zeros(x1_train.shape[0])
        txts_recon_score=np.zeros(x2_train.shape[0])

        txts_recon_score[all_one_indices.cpu().numpy()] = normalized_complete_txts_recon_score
        txts_recon_score[img_missing_idx_eval.cpu().numpy()] = normalized_v2_v1missing_recon_error
        txts_recon_score[txt_missing_idx_eval.cpu().numpy()] = v2_missing_samev1_reconerror

        imgs_recon_score[all_one_indices.cpu().numpy()] = normalized_complete_imgs_recon_score
        imgs_recon_score[txt_missing_idx_eval.cpu().numpy()] = normalized_v1_v2missing_recon_error
        imgs_recon_score[img_missing_idx_eval.cpu().numpy()] = v1_missing_samev2_reconerror

       
            
        if is_weight:
            
            if reverse_weight:
                weights=np.exp(-np.abs(imgs_recon_score-txts_recon_score))
            else:
                weights=np.exp(np.abs(imgs_recon_score-txts_recon_score))
            
            attr_scores=torch.from_numpy(weights).to(device)
        else:
            attr_scores=imgs_recon_score+txts_recon_score
        
        return attr_scores



    def compute_weights(self,common_v1, smp_v1, k):
            weights = []
            for smp in smp_v1:
                knn_indices = self.find_knn(common_v1, smp, k)
                knn_samples = common_v1[knn_indices]
                weight = np.dot(knn_samples, smp)
                weights.append(weight)
            weights=np.array(weights)
            weights=weights / np.sum(weights) 
            return weights

    def fill_missing_data(self,common_v1, common_v2, smp_v1, k):
        
        weights = self.compute_weights(common_v1, smp_v1, k)
        filled_data = []
        for i, smp in enumerate(smp_v1):
            knn_indices = self.find_knn(common_v1, smp, k)
            knn_samples_v2 = common_v2[knn_indices]
            filled = np.dot(weights[i], knn_samples_v2)
            filled_data.append(filled)
        return np.array(filled_data)
    
    def find_knn(self,matrix, target, k):
        distances = np.linalg.norm(matrix - target, axis=1)
        knn_indices = np.argsort(distances)[:k]
        return knn_indices

    def comp_orgin_knn(self,v1,v2,mask,outlabel):
        v1=v1.cpu().numpy()
        v2=v2.cpu().numpy()
        mask=mask.cpu().numpy()



        v2flag = mask[:, 0] == 0
        v1flag = mask[:, 1] == 0
        
        comflag = (mask[:, 1] + mask[:, 0]) == 2

        commonv1=v1[comflag]
        commonv2=v2[comflag]
        comlabel=outlabel[comflag]
        smpv1=v1[v1flag]
        smpv1label=outlabel[v1flag]
        smpv2=v2[v2flag]
        smpv2label=outlabel[v2flag]

        filled_datav1=self.fill_missing_data(commonv2,commonv1,smpv2,k=5)
        filled_datav2=self.fill_missing_data(commonv1,commonv2,smpv1,k=5)

        comv1=np.concatenate((commonv1,smpv1,filled_datav1),axis=0)
        comv2=np.concatenate((commonv2,smpv2,filled_datav2),axis=0)

        labels=np.concatenate((comlabel,smpv1label,smpv2label),axis=0)

        comv1=torch.from_numpy(comv1)
        comv2=torch.from_numpy(comv2)
        return comv1,comv2,labels


        
    def batch_process_autoencoder(self,Total_tensor, batch_size, ae,latent_dim,origin_dim,device):
        """
        Process a tensor using an autoencoder in batches.

        Parameters:
        - Total_tensor: The tensor to be processed
        - batch_size: The size of each batch
        - ae: The autoencoder model

        Returns:
        - concatenated_encoded: The concatenated encoded output
        - concatenated_output: The concatenated decoded output
        """
        ae.eval()
        total = len(Total_tensor) // batch_size + (1 if len(Total_tensor) % batch_size != 0 else 0)
        concatenated_encoded = torch.empty(0,latent_dim).to(device)
        concatenated_output = torch.empty(0,origin_dim).to(device)

        for i in range(total):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            end_idx = min(len(Total_tensor), end_idx)

            batch = Total_tensor[start_idx: end_idx, ...]

            
            encoded_batch = ae.encoder(batch)

            
            output_batch = ae.decoder(encoded_batch)

            if i == 0:
                concatenated_encoded = encoded_batch
                concatenated_output = output_batch
            else:
                concatenated_encoded = torch.cat((concatenated_encoded, encoded_batch), dim=0)
                concatenated_output = torch.cat((concatenated_output, output_batch), dim=0)


        ae.train()
        return concatenated_encoded, concatenated_output
    
    def batch_process_decoder(self,Total_tensor, batch_size, ae,origin_dim,device):
        """
        Process a tensor using an autoencoder in batches.

        Parameters:
        - Total_tensor: The tensor to be processed
        - batch_size: The size of each batch
        - ae: The autoencoder model

        Returns:
        - concatenated_encoded: The concatenated encoded output
        - concatenated_output: The concatenated decoded output
        """
        total = len(Total_tensor) // batch_size + (1 if len(Total_tensor) % batch_size != 0 else 0)
        concatenated_output = torch.empty(0,origin_dim).to(device)

        for i in range(total):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            end_idx = min(len(Total_tensor), end_idx)

            batch = Total_tensor[start_idx: end_idx, ...]


            output_batch = ae.decoder(batch)

            if i == 0:
                concatenated_output = output_batch
            else:
                concatenated_output = torch.cat((concatenated_output, output_batch), dim=0)



        return concatenated_output

def norm_and_cut(tensor1, tensor2, dim=-1):
    
    
    concatenated = torch.cat([tensor1, tensor2], dim=0)
    
    
    normalized = F.normalize(concatenated, dim=dim)
    
    
    norm_tensor1 = normalized[:tensor1.shape[0]]
    norm_tensor2 = normalized[tensor1.shape[0]:]
    
    return norm_tensor1, norm_tensor2
