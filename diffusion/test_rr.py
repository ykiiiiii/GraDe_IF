import torch
import numpy as np
from ema_pytorch import EMA
from gradeif import EGNN_NET,GraDe_IF
import torch.nn.functional as F
import os
from dataset_src.large_dataset import Cath
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def seq_recovery(data,pred_seq):
    '''
    data.x is nature sequence

    '''
    ind = (data.x.argmax(dim=1) == pred_seq.argmax(dim=1))
    recovery = ind.sum()/ind.shape[0]
    return recovery,ind.cpu()

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = 'BLOSUM_3M_small.pt'
ckpt = torch.load('diffusion/results/weight/'+ckpt_path, map_location=device)
config = ckpt['config']

print('noise type',config['noise_type'])
test_dataset_path = 'dataset/CATH42/process/test/'
test_ID = os.listdir(test_dataset_path)
print('Test dataset: ',len(test_ID))
test_dataset = Cath(test_ID,test_dataset_path)
test_loader = DataLoader(test_dataset,batch_size=300,shuffle=False, pin_memory = True, num_workers = 6)
input_feat_dim = test_dataset[0].x.shape[1]+test_dataset[0].extra_x.shape[1]
edge_attr_dim = test_dataset[0].edge_attr.shape[1]

gnn = EGNN_NET(input_feat_dim=config['input_feat_dim'],hidden_channels=config['hidden_dim'],edge_attr_dim=config['edge_attr_dim'],dropout=config['drop_out'],n_layers=config['depth'],update_edge = True,embedding=config['embedding'],embedding_dim=config['embedding_dim'],embed_ss=config['embed_ss'],norm_feat=config['norm_feat'])

diffusion = GraDe_IF(model = gnn,config=config)
diffusion = EMA(diffusion)
diffusion.load_state_dict(ckpt['ema'])
diffusion = diffusion.to(device)

ensemble_list = []
for i in range(50):
    with torch.no_grad():
        ind_all = torch.tensor([])
        all_prob = torch.tensor([])
        all_seq = torch.tensor([])
        for data in test_loader:
            data = data.to(device)
            prob,sample_graph = diffusion.ema_model.ddim_sample(data,diverse=True,step=250)
            recovery, ind = seq_recovery(data,sample_graph)
            ind_all = torch.cat([ind_all,ind])
            all_prob = torch.cat([all_prob,prob.cpu()])
            all_seq = torch.cat([all_seq,data.x.cpu()])

    rr = (ind_all.sum()/ind_all.shape[0]).item()
    ll_fullseq = F.cross_entropy(all_prob,all_seq, reduction='mean').item()
    perplexity = np.exp(ll_fullseq)

    ensemble_list.append(all_prob)
    if i > 0:
        ensemble_prob = torch.stack(ensemble_list).mean(dim = 0)
        ensemble_result = ensemble_prob.argmax(dim=1)
        ensemble_rr = (ensemble_result == all_seq.argmax(dim=1)).sum()/ensemble_result.shape[0]
        print(i,'ensemble rr: ',round(ensemble_rr.item(),4))
        ll_fullseq = F.cross_entropy(ensemble_prob,all_seq, reduction='mean').item()
        perplexity = np.exp(ll_fullseq)
        print(i,'ensemble perplexity: ',round(perplexity,2))
    else:
        print(i,'rr: ',round(rr,4))
        print(i,'perplexity: ',round(perplexity,2))        

print('Final result: ', 'ensemble_rr', round(ensemble_rr.item(),4),'ensemble_perplexity' ,round(perplexity,2), 'one sample rr', round(rr,4))


#Blosum diverse mode
#500 0.5341, 4.02  single sample rr: 0.505
#250 0.5370, 4.06  single sample rr: 0.4679
#100 0.5356, 4.98, single sample rr: 0.4213
#50  0.4827, 8.02, single sample rr: 0.3745

#Blosum non-diverse mode
#500 0.5342, 4.02  single sample rr: 0.505
#250 0.5373, 4.12  single sample rr: 0.4741
#100 0.5351, 7.43, single sample rr: 0.5016
#50  0.4999,16.74, single sample rr: 0.4736

#uniform diverse mode
#500 0.5286 4.08, single sample rr: 0.5022
#250 0.5292 4.13, single sample rr: 0.4325
#100 0.5329 5.28, single sample rr: 0.4222
#50  0.5341 5.91, single sample rr: 0.4212

#uniform non-diverse mode
#500 0.5286 4.08, single sample rr: 0.5022
#250 0.5273 4.09, single sample rr: 0.4357
#100 0.5338 9.49, single sample rr: 0.5095
#50  0.5285 15.53, single sample rr: 0.5113
