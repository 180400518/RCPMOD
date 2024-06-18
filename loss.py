import sys
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def mask_correlated_samples(N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

def mask_correlated_samples_kth( N,othermasksv1,othermasksv2,kth):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        

        for i in range(N//2):
            mask[i,N//2 +othermasksv1[kth]] = 0
            mask[N//2 +othermasksv1[kth], i] = 0

        mask = mask.bool()
        return mask


def contrastive_loss(h_i, h_j,batch_size,contr_weights,contr_weights_v2=None,Kinfo_weight=False,temperature_f=0.5,Onlyposwt=False,Using_wt=True):
    torch.set_num_threads(5) 
    if Using_wt:
        if not Kinfo_weight and not Onlyposwt:
            
            
            h_i=h_i*contr_weights.view(-1,1)
            h_j=h_j*contr_weights.view(-1,1)

        else:
            h_i=h_i*contr_weights.view(-1,1)
            h_j=h_j*contr_weights_v2.view(-1,1)

    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)
    sim = torch.matmul(h, h.T) / temperature_f
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    if Onlyposwt:
        
        
        contr_weights=torch.cat((contr_weights.view(-1,1),contr_weights.view(-1,1)),dim=0)
        positive_samples=positive_samples*contr_weights    
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion=nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N
    
    return loss



def contrastive_loss_oa(h_i, h_j,memoryh_i,memoryh_j,batch_size,contr_weights,contr_weights_v2=None,Kinfo_weight=False,temperature_f=0.5):
    torch.set_num_threads(8) 

    if not Kinfo_weight:
        h_i=h_i*contr_weights.view(-1,1)
        h_j=h_j*contr_weights.view(-1,1)
    else:
        h_i=h_i*contr_weights.view(-1,1)
        h_j=h_j*contr_weights_v2.view(-1,1)


    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)
    memoryh=torch.cat((memoryh_i, memoryh_j), dim=0)
    negsims=torch.matmul(h, memoryh.T) / temperature_f


    sim = torch.matmul(h, h.T) / temperature_f
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    negative_samples=torch.cat((negative_samples,negsims),dim=1)


    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion=nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N
    
    return loss


def contrastive_score(h_i, h_j,batch_size,temperature_f=0.5):
    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)

    sim = torch.matmul(h, h.T) / temperature_f
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion=nn.CrossEntropyLoss(reduction="none")
    scores = criterion(logits, labels)
    scores= (scores[:batch_size] + scores[batch_size:]) / 2
    return scores


def triplet_loss(ins,pos,neg):
    pdist = nn.PairwiseDistance(2)
    per_point_loss = pdist(ins, pos) - pdist(ins, neg)
    per_point_loss = F.relu(per_point_loss)
    loss_triplet = per_point_loss.mean()
    
    return loss_triplet

def uniform_loss(x):
    pdist = nn.PairwiseDistance(2)
    I=pairwise_NNs_inner(x.data)
    distances = pdist(x, x[I])
    loss_uniform = - torch.log(x.shape[0] * distances).mean()  
    return loss_uniform  
                   
def uniform_loss_save(x):
    pdist = nn.PairwiseDistance(2)
    I=pairwise_NNs_inner(x.data)
    distances = pdist(x, x[I])
    loss_uniform = - torch.log(x.shape[0] * distances)
    return loss_uniform 


def pairwise_NNs_inner(x):

    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  
    _, I = torch.max(dots, 1)  
    return I

