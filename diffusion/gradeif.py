import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam,AdamW
import torch_geometric
from torch_geometric.data import Batch,Data
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel

from tqdm.auto import tqdm
from ema_pytorch import EMA

from utils import PredefinedNoiseScheduleDiscrete
from model.egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse
from model.egnn_pytorch.utils import nodeEncoder,edgeEncoder
from dataset_src.large_dataset import Cath

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor<0).any()

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class EGNN_NET(torch.nn.Module):
    def __init__(self, input_feat_dim, hidden_channels, edge_attr_dim,  dropout=0.0, n_layers=1, output_dim = 20,
                 embedding=False, embedding_dim=64, mlp_num=2,update_edge = True,embed_ss = -1,norm_feat = False):
        super(EGNN_NET, self).__init__()
        torch.manual_seed(12345)
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        
        self.embedding = embedding
        self.embed_ss = embed_ss
        self.n_layers = n_layers
        if embedding:
            self.time_mlp = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(),
                                       nn.Linear(hidden_channels, embedding_dim))   

            self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(),
                                        nn.Linear(hidden_channels, embedding_dim))     
        else:
            self.time_mlp = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(),
                                       nn.Linear(hidden_channels, input_feat_dim))
        
            self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(),
                                        nn.Linear(hidden_channels, input_feat_dim))
            

        for i in range(n_layers):
            if embedding:
                layer = EGNN_Sparse(embedding_dim, m_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    mlp_num=mlp_num,update_edge = self.update_edge,norm_feats=norm_feat)
            else:
                layer = EGNN_Sparse(input_feat_dim, m_dim=hidden_channels, edge_attr_dim=edge_attr_dim, dropout=dropout,
                                    mlp_num=mlp_num,update_edge = self.update_edge,norm_feats=norm_feat)
            self.mpnn_layes.append(layer)

            if embedding:
                time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, (embedding_dim) * 2))
                ff_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Dropout(p=dropout),nn.SiLU(), torch_geometric.nn.norm.LayerNorm(embedding_dim),nn.Linear(embedding_dim, embedding_dim)) 
            else:
                time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(input_feat_dim, (input_feat_dim) * 2))
                ff_layer = nn.Sequential(nn.Linear(input_feat_dim, input_feat_dim), nn.Dropout(p=dropout) ,nn.SiLU(), torch_geometric.nn.norm.LayerNorm(input_feat_dim), nn.Linear(input_feat_dim, input_feat_dim)) 

            self.time_mlp_list.append(time_mlp_layer)
            self.ff_list.append(ff_layer)


        if embedding:
            self.node_embedding = nodeEncoder(embedding_dim)
            self.edge_embedding = edgeEncoder(embedding_dim)
            self.lin = Linear(embedding_dim, output_dim)
        else:
            self.lin = Linear(input_feat_dim, output_dim)

    def forward(self, data,time): 
        #data.x first 20 dim is noise label. 21 to 34 is knowledge from backbone, e.g. mu_r_norm, sasa, b factor and so on

        x, pos, extra_x, edge_index, edge_attr,ss, batch = data.x, data.pos, data.extra_x, data.edge_index, data.edge_attr, data.ss,data.batch

        t = self.time_mlp(time)
        ss_embed = self.ss_mlp(ss)
        
        x = torch.cat([x,extra_x],dim=1)
        if self.embedding:
            x = self.node_embedding(x)
            edge_attr = self.edge_embedding(edge_attr)

        x = torch.cat([pos, x], dim=1)

        for i, layer in enumerate(self.mpnn_layes):

            if self.embed_ss == -2 and i == self.n_layers-1:
                corr, feats = x[:,0:3],x[:,3:]
                feats = feats+ss_embed #[N,hidden_dim]+[N,hidden_dim]
                x = torch.cat([corr, feats], dim=-1)

            if self.update_edge:
                h,edge_attr = layer(x, edge_index, edge_attr, batch) #[N,hidden_dim]
            else:
                h = layer(x, edge_index, edge_attr, batch) #[N,hidden_dim]
            
            corr, feats = h[:,0:3],h[:,3:]
            time_emb = self.time_mlp_list[i](t) #[B,hidden_dim*2]
            scale_, shift_ = time_emb.chunk(2,dim=1)
            scale = scale_[data.batch]
            shift = shift_[data.batch]
            feats = feats*(scale+1) +shift
            
            feats = self.ff_list[i](feats)

            x = torch.cat([corr, feats], dim=-1)

        corr, x = x[:,0:3],x[:,3:]
        
        if self.embed_ss == -1:
            x = x+ss_embed 

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

class DiscreteUniformTransition:
    def __init__(self, x_classes: int):
        self.X_classes = x_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes


    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)

        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x

class BlosumTransition:
    def __init__(self, blosum_path='dataset_src/blosum_substitute.pt',x_classes=20,timestep = 500):
        try:
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        except FileNotFoundError:
            blosum_path = '../'+blosum_path
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        self.X_classes = x_classes
        self.timestep = timestep
        temperature_list = self.temperature_list.unsqueeze(dim=0)
        temperature_list = temperature_list.unsqueeze(dim=0)
        Qt_temperature = self.Qt_temperature.unsqueeze(dim=0)
        Qt_temperature = Qt_temperature.unsqueeze(dim=0)
        if temperature_list.shape[0] != self.timestep:
            output_tensor = F.interpolate(temperature_list, size=timestep+1, mode='linear', align_corners=True)
            self.temperature_list = output_tensor.squeeze()
            output_tensor = F.interpolate(Qt_temperature, size=timestep+1, mode='linear', align_corners=True)
            self.Qt_temperature = output_tensor.squeeze()
        else:    
            self.temperature_list = self.temperature_list
            self.Qt_temperature = self.Qt_temperature
    
    def get_Qt_bar(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.temperature_list = self.temperature_list.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.temperature_list[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        q_x[q_x < 1e-6] = 1e-6
        return q_x

    def get_Qt(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.Qt_temperature = self.Qt_temperature.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.Qt_temperature[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        return q_x

class GraDe_IF(nn.Module):
    def __init__(self,model,*,timesteps=500,sampling_timesteps = None,loss_type='CE',objective = 'pred_x0',config = {'noise_type':'uniform'},schedule_fn_kwargs = dict()):
        super().__init__()
        self.model = model
        # self.self_condition = self.model.self_condition
        self.objective = objective
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.transition_model = DiscreteUniformTransition(x_classes=20)
        self.config  = config
        if config['noise_type'] == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=20)
        elif config['noise_type'] == 'blosum':
            self.transition_model = BlosumTransition(timestep=self.timesteps+1)

        assert objective in {'pred_noise', 'pred_x0'}

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine',timesteps=self.timesteps,noise_type='uniform')

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'CE':
            return F.cross_entropy

    def apply_noise(self,data,t_int):
        t_float = t_int / self.timesteps
        if self.config['noise_type'] == 'uniform':
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t_float, device=data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:20].unsqueeze(2)).squeeze()
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t,num_classes = 20)
        noise_data = data.clone()
        noise_data.x = noise_X
        return noise_data

    
    def sample_discrete_feature_noise(self,limit_dist ,num_node):
        x_limit = limit_dist[None,:].expand(num_node,-1) #[num_node,20]
        U_X = x_limit.flatten(end_dim=-2).multinomial(1).squeeze()
        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        return U_X


    def diffusion_loss(self,data,t_int):
        '''
        Compute the divergence between  q(x_t-1|x_t,x_0) and p_{\theta}(x_t-1|x_t)
        
        '''
        # q(x_t-1|x_t,x_0)
        s_int = t_int - 1 
        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps    
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=data.x.device)
        Qt = self.transition_model.get_Qt(beta_t, data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:20].unsqueeze(2)).squeeze()       
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t,num_classes = 20).type_as(data.x)
        prob_true = self.compute_posterior_distribution(noise_X,Qt,Qsb,Qtb,data)  #[N,d_t-1]


        #p_{\theta}(x_t-1|x_t) = \sum_{x0} q(x_t-1|x_t,x_0)p(x0|xt)
        noise_data = data.clone()
        noise_data.x = noise_X #x_t
        t = t_int*torch.ones(size=(data.batch[-1]+1, 1), device=data.x.device).float()
        pred = self.model(noise_data,t)
        pred_X = F.softmax(pred,dim = -1) #\hat{p(X)}_0
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=noise_X,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_pred = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        loss = self.loss_fn(prob_pred,prob_true,reduction='mean')
        return loss

    def compute_val_loss(self,data,evaluate_all=False):
        t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1]+1, 1), device=data.x.device).float()
        diffusion_loss = self.diffusion_loss(data,t_int)
        return diffusion_loss
    
    def compute_batched_over0_posterior_distribution(self,X_t,Q_t,Qsb,Qtb,data):
        """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
        """
        #X_t is a sample of q(x_t|x_t+1)
        Qt_T = Q_t.transpose(-1,-2)
        X_t_ = X_t.unsqueeze(dim = -2)
        left_term = X_t_ @ Qt_T[data.batch] #[N,1,d_t-1]
        # left_term = left_term.unsqueeze(dim = 1) #[N,1,dt-1]

        right_term = Qsb[data.batch] #[N,d0,d_t-1]

        numerator = left_term * right_term #[N,d0,d_t-1]

        prod = Qtb[data.batch] @ X_t.unsqueeze(dim=2) # N,d0,1
        denominator = prod
        denominator[denominator == 0] = 1e-6        

        out = numerator/denominator

        return out

    def compute_posterior_distribution(self,M_t, Qt_M, Qsb_M, Qtb_M,data):
        """ 
        M: is the distribution of X_0
        Compute  q(x_t-1|x_t,x_0) = xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
        """
         
        #X_t is a sample of q(x_t|x_t+1)
        Qt_T = Qt_M.transpose(-1,-2)
        X_t = M_t.unsqueeze(dim = -2)
        left_term = X_t @ Qt_T[data.batch] #[N,1,d_t-1]
        
        M_0 = data.x.unsqueeze(dim = -2) #[N,1,d_t-1]
        right_term = M_0@Qsb_M[data.batch] #[N,1,dt-1]
        numerator = (left_term * right_term).squeeze() #[N,d_t-1]


        X_t_T = M_t.unsqueeze(dim = -1)
        prod = M_0@Qtb_M[data.batch]@X_t_T # [N,1,1]
        denominator = prod.squeeze()
        denominator[denominator == 0] = 1e-6        

        out = (numerator/denominator.unsqueeze(dim=-1)).squeeze()

        return out        #[N,d_t-1]
    
    def sample_p_zs_given_zt(self,t,s,zt,data,cond,diverse,step,last_step):
        """
        sample zs~p(zs|zt)
        """
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        if self.config['noise_type'] == 'uniform':
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, data.x.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t, data.x.device)
            Qsb = self.transition_model.get_Qt_bar(s, data.x.device)

        Qt = (Qsb/Qtb)/(Qsb/Qtb).sum(dim=-1).unsqueeze(dim=2) #approximate

        noise_data = data.clone()
        noise_data.x = zt 
        pred = self.model(noise_data,t*self.timesteps)
        pred_X = F.softmax(pred,dim = -1) 
        
        if isinstance(cond, torch.Tensor):
            pred_X[cond] = data.x[cond]

        if last_step:
            pred_X = F.softmax(pred,dim = -1)
            sample_s = pred_X.argmax(dim = 1)
            final_predicted_X = F.one_hot(sample_s,num_classes = 20).float()

            return pred,final_predicted_X
            
        
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=zt,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20 approximate Q_t-s with Qt 
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        
        if diverse :
            sample_s = prob_X.multinomial(1).squeeze()
        else:
            sample_s = prob_X.argmax(dim=1).squeeze()

        X_s = F.one_hot(sample_s,num_classes = 20).float()

        return X_s,final_predicted_X if last_step else None
    
    def sample(self,data,cond = False,temperature=1.0,stop = 0):
        limit_dist = torch.ones(20)/20
        zt = self.sample_discrete_feature_noise(limit_dist = limit_dist,num_node = data.x.shape[0]) #[N,20] one hot 
        zt = zt.to(data.x.device)
        for s_int in tqdm(list(reversed(range(stop, self.timesteps)))): #500
            #z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
            t_array = s_array + 1
            s_norm = s_array / self.timesteps
            t_norm = t_array /self.timesteps
            zt , final_predicted_X  = self.sample_p_zs_given_zt(t_norm, s_norm,zt, data,cond,temperature,last_step=s_int==stop)
        return zt,final_predicted_X
    
    def ddim_sample(self,data,cond = False,diverse=False,stop = 0,step=50):
        limit_dist = torch.ones(20)/20
        zt = self.sample_discrete_feature_noise(limit_dist = limit_dist,num_node = data.x.shape[0]) #[N,20] one hot 
        zt = zt.to(data.x.device)
        for s_int in tqdm(list(reversed(range(stop, self.timesteps,step)))): #500
            #z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
            t_array = s_array + step
            s_norm = s_array / self.timesteps
            t_norm = t_array /self.timesteps
            zt , final_predicted_X  = self.sample_p_zs_given_zt(t_norm, s_norm,zt, data,cond,diverse,step,last_step=s_int==stop)
        return zt,final_predicted_X


    def forward(self,data,logit=False):
        t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1]+1, 1), device=data.x.device).float()
        noise_data = self.apply_noise(data ,t_int)
        pred_X = self.model(noise_data,t_int) #have parameter

        if self.objective == 'pred_x0':
            target = data.x
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(pred_X,target,reduction='mean')
        if logit:
            return loss, pred_X
        else:
            return loss

def seq_recovery(data,pred_seq):
    '''
    data.x is nature sequence

    '''
    ind = (data.x.argmax(dim=1) == pred_seq.argmax(dim=1))
    recovery = ind.sum()/ind.shape[0]
    return recovery,ind.cpu()


class Trianer(object):
    def __init__(
        self,
        config,
        diffusion_model,
        train_dataset,
        val_dataset,
        test_dataset,
        *,
        train_batch_size = 512,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        weight_decay = 1e-2,
        train_num_steps = 200000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 2,
        num_samples = 25,
        results_folder = './diffusion/results',
    ):    
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = diffusion_model.to(device)
        self.config = config
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        self.ds = train_dataset
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 6)

        self.dl = cycle(dl)

        self.val_loader = DataLoader(val_dataset,batch_size=train_batch_size,shuffle=False, pin_memory = True, num_workers = 6)
        self.test_loader = DataLoader(test_dataset,batch_size=train_batch_size,shuffle=False, pin_memory = True, num_workers = 6)
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas,weight_decay=weight_decay)
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        Path(results_folder+'/weight/').mkdir(exist_ok = True)
        Path(results_folder+'/figure/').mkdir(exist_ok = True)

        self.step = 0
        self.save_file_name = self.config['Date']+f"_result_lr={self.config['lr']}_dp={self.config['drop_out']}_clip={self.config['clip_grad_norm']}_timestep={self.config['timesteps']}_depth={self.config['depth']}_hidden={self.config['hidden_dim']}_embedding={self.config['embedding']}_embed_dim={self.config['embedding_dim']}_ss={self.config['embed_ss']}_noise={self.config['noise_type']}"
    def save(self, milestone):

        data = {
            'config': self.config,
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }

        torch.save(data, os.path.join(str(self.results_folder),'weight', self.save_file_name+f'_{milestone//((len(self.ds)//self.batch_size))}.pt'))
    
    def load(self, milestone,filename =False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if filename:
            data = torch.load(str(self.results_folder)+'/'+filename, map_location=device)
        else:
            data = torch.load(str(self.results_folder / self.config['Date']+f"model_lr={self.config['lr']}_dp={self.config['drop_out']}_timestep={self.config['timesteps']}_hidden={self.config['hidden_dim']}_{milestone}.pt"), map_location=device)

        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")


    def train(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loss,recovery_list,perplexity, total_loss,val_loss_list= [],[],[],0,[]
        val_loss = torch.tensor([5.0])
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:


                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)


                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                all_iter = len(self.ds)//self.batch_size
                num_iter = self.step % all_iter +1
                pbar.set_description(f'loss: {total_loss/num_iter:.4f}')

                if self.step%(len(self.ds)//self.batch_size) == 0:
                    
                    train_loss.append(total_loss/all_iter)

                    val_loss_list.append(val_loss.item())
                    total_loss = 0

                self.opt.step()
                self.opt.zero_grad()


                self.step += 1
                if self.step > self.train_num_steps/2:
                    for g in self.opt.param_groups:
                        g['lr'] = self.config['lr']*0.1
                        
                self.ema.to(device)
                self.ema.update()
                if self.step != 0 and self.step % (self.save_and_sample_every*(len(self.ds)//self.batch_size)) == 0:
                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        ind_all = torch.tensor([])
                        all_prob = torch.tensor([])
                        all_seq = torch.tensor([])
                        for data in self.val_loader:
                            data = data.to(device)
                            val_loss = self.ema.ema_model.compute_val_loss(data,False)
                            prob,sample_graph = self.ema.ema_model.ddim_sample(data,diverse = True,step=100) #zt is the output of Neural Netowrk and sample graph is a sample of it
                            _, ind = seq_recovery(data,sample_graph)
                            ind_all = torch.cat([ind_all,ind])
                            all_prob = torch.cat([all_prob,prob.cpu()])
                            all_seq = torch.cat([all_seq,data.x.cpu()])

                        milestone = self.step // self.save_and_sample_every

                    recovery_list.append((ind_all.sum()/ind_all.shape[0]).item())
                    ll_fullseq = F.cross_entropy(all_prob,all_seq, reduction='mean').item()
                    perplexity.append(np.exp(ll_fullseq)*0.01) #for the same scale
                    print(f'recovery rate is {recovery_list[-1]}')
                    print(f'perplexity : {perplexity[-1]:.2f}')
                    fig, axs = plt.subplots(2,1, figsize=(10, 5))
                    axs[0].plot(train_loss,label = 'train_loss')
                    axs[0].plot(val_loss_list,label = 'val_loss')
                    axs[0].set_ylim((0,5))

                    axs[1].plot(recovery_list,label = 'recovery')
                    axs[1].plot(perplexity,label = 'perplexity')
                    axs[0].legend(loc="upper right", fancybox=True)
                    axs[0].set_title(f'best_recovery={max(recovery_list):.4f}')

                    plt.savefig(os.path.join(str(self.results_folder),'figure', self.save_file_name+f'.png'),dpi = 200)
                    plt.close()

                    if recovery_list[-1] == max(recovery_list):
                        self.save(milestone)

                pbar.update(1)

        print('training complete')    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--Date', type = str,default='Mar_2th',
                        help='Date of experiment')
    
    parser.add_argument('--train_dir', type = str,default='dataset/process/train/',
                        help='path of training data')
    
    parser.add_argument('--val_dir', type = str,default='dataset/process/validation/',
                        help='path of val data')
    
    parser.add_argument('--test_dir', type = str,default='dataset/process/test/',
                        help='path of test data')  

    parser.add_argument('--ts_train_dir', type = str,default='dataset/TS/training_set/process/',
                        help='path of training data') 

    parser.add_argument('--ts_test_dir', type = str,default='dataset/TS/test_set/T500/process/',
                        help='path of test data')    
     
    parser.add_argument('--objective', type = str,default='pred_x0',
                        help='the target of training objective, objective must be either pred_x0 or smooth_x0')    

    parser.add_argument('--dataset', type = str,default='CATH',help='the dataset to train on')

    parser.add_argument('--lr', type = float,default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--wd', type = float,default=1e-2,
                        help='weight decay')

    parser.add_argument('--drop_out', type = float,default=0.0,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--timesteps',  type = int,default=500,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--hidden_dim',  type = int,default=256,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--device_id', type = int,default=0,
                        help='cuda device')

    parser.add_argument('--batch_size', type = int,default=64,help='batch_size')
    
    parser.add_argument('--ema_decay', type = float,default=0.995,help='ema_decay')
    
    parser.add_argument('--depth', type = int,default=1,
                        help='number of GNN layers')
    
    parser.add_argument('--embedding_dim', type = int,default=16,
                        help='the dim of feature embedding')   
    
    parser.add_argument('--clip_grad_norm',  type = float,default=1.0,
                        help='clip_grad_norm')
    
    parser.add_argument('--embedding', action='store_true',#default = False,
                        help='whether residual embedding the feature') 
    
    parser.add_argument('--norm_feat', action='store_true',#default = False,
                        help='whether normalization node feature in egnn')  

    parser.add_argument('--updeate_edge', action='store_false',help='whether update edge feature in egnn')
    
    parser.add_argument('--embed_ss', type = int,default=-1,
                        help='when add ss embedding into gnn') 
    
    parser.add_argument('--noise_type', type = str,default='uniform',help='the type of noise ,uniform or blosum')
    args = parser.parse_args()

    config = vars(args)

    if config['dataset'] == 'CATH':
        print('train on CATH dataset')
        train_ID ,val_ID,test_ID= os.listdir(config['train_dir']),os.listdir(config['val_dir']),os.listdir(config['test_dir'])
        train_dataset = Cath(train_ID,config['train_dir'])
        val_dataset = Cath(val_ID,config['val_dir'])
        test_dataset = Cath(test_ID,config['test_dir'])
        print(f'train on CATH dataset with {len(train_dataset)}  training data and {len(val_dataset)}  val data')

    elif config['dataset'] == 'TS':
        basedir = config['train_dir']
        train_ID ,val_ID= os.listdir(config['ts_train_dir']),os.listdir(config['ts_test_dir'])
        train_dataset = Cath(train_ID,config['ts_train_dir'])#+数据
        val_dataset = Cath(val_ID,config['ts_test_dir'])
        test_dataset = Cath(val_ID,config['ts_test_dir'])
        print(f'train on TS dataset with {len(train_dataset)}  training data and {len(val_dataset)}  val data')        
    else:
        raise ValueError(f"unknown dataset") 

    input_feat_dim = train_dataset[0].x.shape[1]+train_dataset[0].extra_x.shape[1]
    edge_attr_dim = train_dataset[0].edge_attr.shape[1]

    config['input_feat_dim'] = input_feat_dim
    config['edge_attr_dim'] = edge_attr_dim

    model = EGNN_NET(input_feat_dim=input_feat_dim,hidden_channels=config['hidden_dim'],edge_attr_dim=edge_attr_dim,dropout=config['drop_out'],n_layers=config['depth'],update_edge = config['updeate_edge'],embedding=config['embedding'],embedding_dim=config['embedding_dim'],norm_feat=config['norm_feat'],embed_ss=config['embed_ss'])
    diffusion_model = GraDe_IF(model,timesteps=config['timesteps'],objective=config['objective'],config=config)
    trainer  = Trianer(config,
                        diffusion_model,
                        train_dataset, 
                        val_dataset,
                        test_dataset,
                        train_batch_size = config['batch_size'],
                        train_lr=config['lr'],
                        weight_decay = config['wd'],
                        ema_decay= config['ema_decay'])
    trainer.train()
