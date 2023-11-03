
from torch_geometric.data import Dataset, download_url,Batch,Data
import torch
import os
from torch_geometric.loader import DataListLoader, DataLoader
import random

class Cath(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, baseDIR,transform=None, pre_transform=None, pre_filter=None,pred_sasa = False):
        super().__init__(baseDIR, transform, pre_transform, pre_filter)
        'Initialization'
        self.list_IDs = list_IDs
        self.baseDIR = baseDIR
        self.pred_sasa = pred_sasa

    def len(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        data = torch.load(self.baseDIR+ID)
        del data['distances']
        del data['edge_dist']
        mu_r_norm=data.mu_r_norm
        extra_x_feature = torch.cat([data.x[:,20:],mu_r_norm],dim = 1)
        graph = Data(
            x=data.x[:, :20],
            extra_x = extra_x_feature,
            pos=data.pos,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ss = data.ss[:data.x.shape[0],:],
            sasa = data.x[:,20]
        )
        return graph
    


if __name__ == '__main__':
    basedir = 'dataset/cath40_k10_imem_add2ndstrc/process/'

    filelist = os.listdir(basedir)
    filelist.sort()
    random.Random(4).shuffle(filelist)
    test_filelist = filelist[-500:]
    test_dataset = Cath(test_filelist,basedir)
    data = test_dataset[10]
    print(data)

    # dl = DataLoader(train_dataset, batch_size = 10, shuffle = True, pin_memory = True, num_workers = 0)
    # for data_list in dl:
    #      print(data_list)
        #  data = Batch.from_data_list(data_list) 
        #  print(data)