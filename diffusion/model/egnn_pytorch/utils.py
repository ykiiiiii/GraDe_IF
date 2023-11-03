import torch
from torch import sin, cos, atan2, acos

def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def get_node_feature_dims():
    '''
    each node has 26 dim feature corrsponding to residual type, sasa, bfactor,dihedral, mu_r_norm

    update Apr 6th:
    remove bfactor as there is no bfactor in predicted structure
    '''
    return [20, 1,1,4, 5]


def get_edge_feature_dims():
    '''
    each node has 93 dim feature corrsponding to one hot sequence distance, interatomic distance, local frame orientation
    '''
    return [65, 1, 15, 12]


class nodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim,feature_num=4):
        super(nodeEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        if feature_num == 4:
            self.node_feature_dim = get_node_feature_dims()
        else:
            self.node_feature_dim = [20, 4, 5]
            
        for i, dim in enumerate(self.node_feature_dim):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.node_feature_dim)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.node_feature_dim[i]])
            feature_dim_count += self.node_feature_dim[i]
        return x_embedding



class edgeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(edgeEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.edge_feature_dims = get_edge_feature_dims()
        for i, dim in enumerate(self.edge_feature_dims):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.edge_feature_dims)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.edge_feature_dims[i]])
            feature_dim_count += self.edge_feature_dims[i]
        return x_embedding