from dataset_src.utils import normalize, safe_index, one_hot_res, log, dihedral, NormalizeProtein, dataset_argument, get_stat
import argparse
from ast import Continue
from json import load
from typing import Callable, List, Optional
from torch_geometric.loader import DataLoader
import os
import torch
import sys
import math
import numpy as np
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from torch_geometric.io import read_txt_array
from tqdm import tqdm
import scipy.spatial as spa
from scipy.special import softmax
from Bio.PDB import get_surface, PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import calc_dihedral
from rdkit.Chem import GetPeriodicTable
import random
import warnings
import torch.nn.functional as F
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from posixpath import split
import torch
import os
import copy
from torch_geometric.data import InMemoryDataset, Data

from collections.abc import Mapping, Sequence
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.data.collate import collate
from torch_geometric.data.dataset import Dataset, IndexType
from torch_geometric.data.separate import separate
import numpy as np
from numpy import argmax





warnings.filterwarnings("ignore")


class Cath_imem(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        raw_dir (string, optional): Root directory where the
        original dataset stored(default: :obj:`None`)

        num_residue_type (int, optional): The number of amino acid types.
        (default: obj:'20')
        micro_radius (int, optional): The radius of micro-environment 
        centered on the mask node. (default: obj:'20')
        c_alpha_max_neighbors (int, optional): The number of maximum 
        connected nodes. (default: obj:'10') 
        cutoff (int, optional): The maximum connected nodes distance 
        (default: obj:'30') 
        seq_dist_cut (int, optional): one-hot encoding the sequence distance
        edge attribute
        (default: obj:)
        [0.25,0.5,0.75,0.9,0.95,0.98,0.99]
        [  2.   3.  13.  63. 127. 247. 347.]
        num_val (int, optional): The number of validation samples in case of "random" split. (default: 500)
        num_test (int, optional): The number of test samples in case of "random" split. (default: 1000)

        use_micro (bool, optional): If :obj:`True`, the dataset will 
        use microenvironment graph.(default: obj:'False')
        use_angle (bool, optional): If :obj:'True', the dataset will
        regard dihedral angles as one of its node attribute. If :obj:'False',
        the dataset will use the cos and sin value of these. (default: obj:'False')
        use_omega (bool,optional): If :obj:'True', the dataset will 
        contain omega dihedral node attribute.
        (default: obj:'False')
        random_sampling (bool,optional):
        (default: obj:'False')
        # use_localdatastet (bool) (bool,optional): If :obj:'True', online dataset 
        # will be downloaded. If not, local pdb files will be used
        # (default: obj:'True')

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    url = 'http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S{}-v4_3_0.pdb.tgz'
    names = ['20', '40']
    splits = ['train', 'val', 'test']
    allowable_features = {
        'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
        'possible_chirality_list': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER'
        ],
        'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
        'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_hybridization_list': [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
        'possible_is_aromatic_list': [False, True],
        'possible_is_in_ring3_list': [False, True],
        'possible_is_in_ring4_list': [False, True],
        'possible_is_in_ring5_list': [False, True],
        'possible_is_in_ring6_list': [False, True],
        'possible_is_in_ring7_list': [False, True],
        'possible_is_in_ring8_list': [False, True],
        'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                                 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                                 'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
        'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                                 'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
        'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                                 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                                 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
    }
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    num_struc_2nds_res_alphabet = len(struc_2nds_res_alphabet)
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))

    def __init__(self, root: str, name: str,
                 split: str = 'train',
                 num_residue_type: int = 20,
                 micro_radius: int = 20,
                 c_alpha_max_neighbors: int = 10,
                 cutoff: int = 30,
                 seq_dist_cut: int = 64,
                 use_micro: bool = False,
                 use_angle: bool = False,
                 use_omega: bool = False,
                 random_sampling: bool = True,
                 diffusion:bool = False,
                 pre_equivariant:bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 divide_num: int = 1,
                 divide_idx: int = 0,
                 set_length: int = 500,
                 num_val: int = 500,
                 num_test: int = 500,
                 is_normalize: bool = True,
                 normalize_file: str = None,
                 struc_2nds_res_path: str = None,
                 ):
        self.name = name
        assert self.name in self.names
        self.split = split
        assert self.split in self.splits

        self.num_residue_type = num_residue_type
        self.micro_radius = micro_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.seq_dist_cut = seq_dist_cut
        self.cutoff = cutoff

        self.num_val = num_val
        self.num_test = num_test
        self.divide_num = divide_num
        self.divide_idx = divide_idx
        self.set_length = set_length

        self.is_normalize = is_normalize
        self.normalize_file = normalize_file

        self.use_micro = use_micro
        self.use_angle = use_angle
        self.use_omega = use_omega
        self.random_sampling = random_sampling
        self.diffusion = diffusion
        self.pre_equivariant = pre_equivariant

        self.wrong_proteins = ['1kp0A01', '2atcA02']

        self.sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
                               n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.
        self.periodic_table = GetPeriodicTable()
        self.biopython_parser = PDBParser()
        self.struc_2nds_res_path = struc_2nds_res_path

        # super().__init__(root, transform, pre_transform, pre_filter)

        # self.data, self.slices = torch.load(
        #     self.processed_paths[self.splits.index(self.split)])
        # self.nums_amino_cum = self.slices['x']

    @property
    def raw_file_names(self) -> str:
        raw_file_names = os.path.join(self.raw_dir, 'dompdb')
        if not os.path.exists(raw_file_names):
            os.mkdir(raw_file_names)
        return raw_file_names

    @property
    def raw_dir(self) -> str:
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        raw_dir = os.path.join(self.root, 'raw')
        if not os.path.exists(raw_dir):
            os.mkdir(raw_dir)
        return raw_dir

    @property
    def saved_graph_dir(self) -> str:
        dir_root = os.path.join(self.root)
        if not os.path.exists(dir_root):
            os.mkdir(dir_root)
        dir_name = os.path.join(dir_root, 'graph')
        dir_name_process = os.path.join(dir_root, 'process')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            os.mkdir(dir_name_process)
        if not self.set_length:
            self.set_length = len(os.listdir(dir_name))
        return dir_name

    @property
    def saved_amino_cum(self) -> str:
        amino_cum_name = os.path.join(
            self.root, 'amino_cum.pt')
        return amino_cum_name

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> str:
        return ['train.pt', 'val.pt',  'test.pt']

    # def download(self):
    #     if len(os.listdir(self.raw_file_names)) > 0:
    #         return 0
    #     path = download_url(self.url.format(self.name), self.raw_dir)
    #     extract_tar(path, self.raw_dir)
    #     os.unlink(path)
        
    def write_info(self):
        written_filename = os.path.join(self.root, 'wrong_protein_names.txt')
        file = open(written_filename, "w+")
        for protein_name in self.wrong_proteins:
            file.writelines(protein_name + '\n')
        file.close()

    def process(self):
        self.generate_protein_graph()
        self.write_info()

        filenames = os.listdir(self.saved_graph_dir)
        filenames.sort()
        protein_length = len(filenames)
        if self.set_length:
            protein_length = min(protein_length, self.set_length)

        if not self.normalize_file:
            self.normalize_file = get_stat(self.saved_graph_dir)

        random.Random(4).shuffle(filenames)
        filenames_list = [filenames[:-(self.num_val + self.num_test)],
                          filenames[-(self.num_val + self.num_test):-self.num_test],
                          filenames[-self.num_test:]]

        for k in range(3):
            data_list = []
            dist_list = []
            print(filenames_list[k])
            for file in filenames_list[k]:
                graph = torch.load(os.path.join(self.saved_graph_dir, file))
                dist_list.append(graph.distances)
                del graph['distances']
                del graph['edge_dist']

                if self.is_normalize:
                    normalize_transform = NormalizeProtein(filename=self.normalize_file)
                    graph = normalize_transform(graph)
                if self.pre_filter is not None:
                    graph =  self.pre_filte(graph)
                    # data_list = [d for d in data_list if self.pre_filter(d)]

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                    # data_list = [self.pre_transform(d) for d in data_list]
                data_list.append(graph)
                torch.save(graph,os.path.join('dataset/cath40_k10_imem_add2ndstrc/process', file))


            self.collate(data_list)
            torch.save(self.collate(data_list), self.processed_paths[k])

    def generate_protein_graph(self):
        names = os.listdir(self.raw_file_names)
        names.sort()
        n = int(np.ceil(len(names) / self.divide_num))
        names = names[n*self.divide_idx:min(len(names), n*(self.divide_idx+1))]
        for idx, name in enumerate(tqdm(names)):
            if name in ['1byiA00','1c1yB00']:
#             if name in ['1bteA00', '1byiA00','1c1yB00']:
                continue
            # print(name)
            saved_graph_filename = os.path.join(
                self.saved_graph_dir, name + '.pt')
            if os.path.exists(saved_graph_filename):
                continue
            protein_filename = os.path.join(self.raw_file_names, name)
            if (name in self.wrong_proteins) or (not protein_filename):
                continue
            rec, rec_coords, c_alpha_coords, n_coords, c_coords = self.get_receptor_inference(protein_filename)
            if self.struc_2nds_res_path:
                struc_2nds_res_filename = os.path.join(self.struc_2nds_res_path, name)
                struc_2nd_res = self.get_struc2ndRes(struc_2nds_res_filename)
            else:
                struc_2nd_res = None
            if struc_2nd_res is not None:
                if c_alpha_coords.shape[0] == struc_2nd_res.shape[0]:
                    rec_graph = self.get_calpha_graph(
                        rec, c_alpha_coords, n_coords, c_coords, rec_coords, struc_2nd_res)
                    if not rec_graph:
                        self.wrong_proteins.append(name)
                        continue
                    # print(rec_graph)
                    torch.save(rec_graph, saved_graph_filename)
                else: 
                    self.wrong_proteins.append(name)
            

    def rec_residue_featurizer(self, rec, one_hot=True, add_feature=None):
        num_res = len([_ for _ in rec.get_residues()])
        num_feature = 2
        for add_feature_1 in add_feature:
            if add_feature_1.any():
                num_feature += add_feature_1.shape[1]
                # print(add_feature_1.shape)
        res_feature = torch.zeros(num_res, self.num_residue_type + num_feature)
        count = 0
        try:
            self.sr.compute(rec, level="R")
            for residue in rec.get_residues():
                sasa = residue.sasa
                for atom in residue:
                    if atom.name == 'CA':
                        bfactor = atom.bfactor
                assert not np.isinf(bfactor)
                assert not np.isnan(bfactor)
                assert not np.isinf(sasa)
                assert not np.isnan(sasa)

                residx = safe_index(
                    self.allowable_features['possible_amino_acids'], residue.get_resname())
                res_feat_1 = one_hot_res(
                    residx, num_residue_type=self.num_residue_type) if one_hot else [residx]
                if not res_feat_1:
                    return False
                res_feat_1.append(sasa)
                res_feat_1.append(bfactor)
                if num_feature > 2:
                    for add_feature_1 in add_feature:
                        if add_feature_1.any():
                            res_feat_1.extend(list(add_feature_1[count, :]))
                res_feature[count, :] = torch.tensor(
                    res_feat_1, dtype=torch.float32)
                count += 1

            for k in range(self.num_residue_type, self.num_residue_type + 2):
                mean = res_feature[:, k].mean()
                std = res_feature[:, k].std()
                res_feature[:, k] = (res_feature[:, k] -
                                    mean) / (std + 0.000000001)
            return res_feature
        except TypeError:
            return None

    def get_node_features(self, n_coords, c_coords, c_alpha_coords, coord_mask, with_coord_mask=True, use_angle=False, use_omega=False):
        num_res = n_coords.shape[0]
        if use_omega:
            num_angle_type = 3
            angles = np.zeros((num_res, num_angle_type))
            for i in range(num_res-1):
                # These angles are called φ (phi) which involves the backbone atoms C-N-Cα-C
                angles[i, 0] = dihedral(
                    c_coords[i], n_coords[i], c_alpha_coords[i], n_coords[i+1])
                # psi involves the backbone atoms N-Cα-C-N.
                angles[i, 1] = dihedral(
                    n_coords[i], c_alpha_coords[i], c_coords[i], n_coords[i+1])
                angles[i, 2] = dihedral(
                    c_alpha_coords[i], c_coords[i], n_coords[i+1], c_alpha_coords[i+1])
        else:
            num_angle_type = 2
            angles = np.zeros((num_res, num_angle_type))
            for i in range(num_res-1):
                # These angles are called φ (phi) which involves the backbone atoms C-N-Cα-C
                angles[i, 0] = dihedral(
                    c_coords[i], n_coords[i], c_alpha_coords[i], n_coords[i+1])
                # psi involves the backbone atoms N-Cα-C-N.
                angles[i, 1] = dihedral(
                    n_coords[i], c_alpha_coords[i], c_coords[i], n_coords[i+1])
        if use_angle:
            node_scalar_features = angles
        else:
            node_scalar_features = np.zeros((num_res, num_angle_type*2))
            for i in range(num_angle_type):
                node_scalar_features[:, 2*i] = np.sin(angles[:, i])
                node_scalar_features[:, 2*i + 1] = np.cos(angles[:, i])

        if with_coord_mask:
            node_scalar_features = torch.cat([
                node_scalar_features,
                coord_mask.float().unsqueeze(-1)
            ], dim=-1)
        node_vector_features = None
        return node_scalar_features, node_vector_features

    def get_calpha_graph(self, rec, c_alpha_coords, n_coords, c_coords, coords, struc_2nd_res = None):
        scalar_feature, vec_feature = self.get_node_features(
            n_coords, c_coords, c_alpha_coords, coord_mask=None, with_coord_mask=False, use_angle=self.use_angle, use_omega=self.use_omega)

        # Extract 3D coordinates and n_i,u_i,v_i
        # vectors of representative residues ################
        residue_representatives_loc_list = []
        n_i_list = []
        u_i_list = []
        v_i_list = []
        for i, residue in enumerate(rec.get_residues()):
            n_coord = n_coords[i]
            c_alpha_coord = c_alpha_coords[i]
            c_coord = c_coords[i]
            u_i = (n_coord - c_alpha_coord) / \
                np.linalg.norm(n_coord - c_alpha_coord)
            t_i = (c_coord - c_alpha_coord) / \
                np.linalg.norm(c_coord - c_alpha_coord)
            n_i = np.cross(u_i, t_i) / \
                np.linalg.norm(np.cross(u_i, t_i))   # main chain
            v_i = np.cross(n_i, u_i)
            assert (math.fabs(
                np.linalg.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
            n_i_list.append(n_i)
            u_i_list.append(u_i)
            v_i_list.append(v_i)
            residue_representatives_loc_list.append(c_alpha_coord)

        residue_representatives_loc_feat = np.stack(
            residue_representatives_loc_list, axis=0)  # (N_res, 3)
        n_i_feat = np.stack(n_i_list, axis=0)
        u_i_feat = np.stack(u_i_list, axis=0)
        v_i_feat = np.stack(v_i_list, axis=0)
        num_residues = len(c_alpha_coords)
        if num_residues <= 1:
            raise ValueError(f"rec contains only 1 residue!")

        ################### Build the k-NN graph ##############################
        assert num_residues == residue_representatives_loc_feat.shape[0]
        assert residue_representatives_loc_feat.shape[1] == 3
        distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)

        src_list = []
        dst_list = []
        dist_list = []
        mean_norm_list = []
        for i in range(num_residues):
            dst = list(np.where(distances[i, :] < self.cutoff)[0])
            dst.remove(i)
            if self.c_alpha_max_neighbors != None and len(dst) > self.c_alpha_max_neighbors:
                dst = list(np.argsort(distances[i, :]))[
                    1: self.c_alpha_max_neighbors + 1]
            if len(dst) == 0:
                # choose second because first is i itself
                dst = list(np.argsort(distances[i, :]))[1:2]
                log(
                    f'The c_alpha_cutoff {self.cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
            valid_dist = list(distances[i, dst])
            dist_list.extend(valid_dist)
            valid_dist_np = distances[i, dst]
            sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            weights = softmax(- valid_dist_np.reshape((1, -1))
                              ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
            assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
            diff_vecs = residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[
                dst, :]  # (neigh_num, 3)
            mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
            denominator = weights.dot(np.linalg.norm(
                diff_vecs, axis=1))  # (sigma_num,)
            mean_vec_ratio_norm = np.linalg.norm(
                mean_vec, axis=1) / denominator  # (sigma_num,)
            mean_norm_list.append(mean_vec_ratio_norm)
        assert len(src_list) == len(dst_list)
        assert len(dist_list) == len(dst_list)
        residue_representatives_loc_feat = torch.from_numpy(
            residue_representatives_loc_feat.astype(np.float32))
        if struc_2nd_res != None:
            x = self.rec_residue_featurizer(
                rec, one_hot=True, add_feature=[scalar_feature,struc_2nd_res])
        else:
            x = self.rec_residue_featurizer(
                rec, one_hot=True, add_feature=[scalar_feature])
            
        if isinstance(x, bool) and (not x):
            return False
        if x is not None:
            graph = Data(
                x=x[:,:-8],
                ss = x[:,-8:],
                pos=residue_representatives_loc_feat,
                edge_attr=self.get_edge_features(
                    src_list, dst_list, dist_list, divisor=4),
                edge_index=torch.tensor([src_list, dst_list]),
                edge_dist=torch.tensor(dist_list),
                distances=torch.tensor(distances),
                mu_r_norm=torch.from_numpy(np.array(mean_norm_list).astype(np.float32)))

            # Loop over all edges of the graph and build the various p_ij, q_ij, k_ij, t_ij pairs
            edge_feat_ori_list = []
            for i in range(len(dist_list)):
                src = src_list[i]
                dst = dst_list[i]
                # place n_i, u_i, v_i as lines in a 3x3 basis matrix
                basis_matrix = np.stack(
                    (n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
                p_ij = np.matmul(basis_matrix,
                                residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[
                                    dst, :])
                q_ij = np.matmul(basis_matrix, n_i_feat[src, :])  # shape (3,)
                k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
                t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
                s_ij = np.concatenate(
                    (p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
                edge_feat_ori_list.append(s_ij)

            edge_feat_ori_feat = np.stack(
                edge_feat_ori_list, axis=0)  # shape (num_edges, 4, 3)
            edge_feat_ori_feat = torch.from_numpy(
                edge_feat_ori_feat.astype(np.float32))

            graph.edge_attr = torch.cat(
                [graph.edge_attr, edge_feat_ori_feat], axis=1)  # (num_edges, 17)

            return graph
        else:
            return None

    def remove_node(self, graph, node_idx):
        new_graph = Data.clone(graph)
        # delete node
        new_graph.x = torch.cat(
            [new_graph.x[:node_idx, :], new_graph.x[node_idx+1:, :]])
        new_graph.ss = torch.cat(
            [new_graph.ss[:node_idx, :], new_graph.ss[node_idx+1:, :]])
        new_graph.pos = torch.cat(
            [new_graph.pos[:node_idx, :], new_graph.pos[node_idx+1:, :]])
        new_graph.mu_r_norm = torch.cat(
            [new_graph.mu_r_norm[:node_idx, :], new_graph.mu_r_norm[node_idx+1:, :]])

        # delete edge
        keep_edge = (torch.sum(new_graph.edge_index == node_idx, dim=0) == 0)
        new_graph.edge_index = new_graph.edge_index[:, keep_edge]
        new_graph.edge_attr = new_graph.edge_attr[keep_edge, :]
        return new_graph

    def get_edge_features(self, src_list, dst_list, dist_list, divisor=4):
        seq_edge = torch.absolute(torch.tensor(
            src_list) - torch.tensor(dst_list)).reshape(-1, 1)
        seq_edge = torch.where(seq_edge > self.seq_dist_cut,
                               self.seq_dist_cut, seq_edge)
        seq_edge = F.one_hot(
            seq_edge, num_classes=self.seq_dist_cut + 1).reshape((-1, self.seq_dist_cut + 1))
        contact_sig = torch.where(torch.tensor(
            dist_list) <= 8, 1, 0).reshape(-1, 1)
        # avg distance = 7. So divisor = (4/7)*7 = 4
        dist_fea = self.distance_featurizer(dist_list, divisor=divisor)
        return torch.concat([seq_edge, dist_fea, contact_sig], dim=-1)
    
    def get_struc2ndRes(self, struc_2nds_res_filename):
        if os.path.isfile(struc_2nds_res_filename):
            #open text file in read mode
            text_file = open(struc_2nds_res_filename, "r")
            #read whole file to a string
            data = text_file.read()
            #close file
            text_file.close()
            # integer encode input data
            integer_encoded = [self.char_to_int[char] for char in data]
            print(len(data))
            data = F.one_hot(torch.tensor(integer_encoded), num_classes = self.num_struc_2nds_res_alphabet)
            return data
        else:
            print('Warning: ' + struc_2nds_res_filename + 'does not exist')
            return None

    def get_receptor_inference(self, rec_path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            structure = self.biopython_parser.get_structure(
                'random_id', rec_path)
            rec = structure[0]
        coords = []
        c_alpha_coords = []
        n_coords = []
        c_coords = []
        valid_chain_ids = []
        lengths = []
        for i, chain in enumerate(rec):
            chain_coords = []  # num_residues, num_atoms, 3
            chain_c_alpha_coords = []
            chain_n_coords = []
            chain_c_coords = []
            count = 0
            invalid_res_ids = []
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    invalid_res_ids.append(residue.get_id())
                    continue
                residue_coords = []
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == 'CA':
                        c_alpha = list(atom.get_vector())
                    if atom.name == 'N':
                        n = list(atom.get_vector())
                    if atom.name == 'C':
                        c = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector()))
                # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                if c_alpha != None and n != None and c != None:
                    chain_c_alpha_coords.append(c_alpha)
                    chain_n_coords.append(n)
                    chain_c_coords.append(c)
                    chain_coords.append(np.array(residue_coords))
                    count += 1
                else:
                    invalid_res_ids.append(residue.get_id())
            for res_id in invalid_res_ids:
                chain.detach_child(res_id)
            lengths.append(count)
            coords.append(chain_coords)
            c_alpha_coords.append(np.array(chain_c_alpha_coords))
            n_coords.append(np.array(chain_n_coords))
            c_coords.append(np.array(chain_c_coords))
            if len(chain_coords) > 0:
                valid_chain_ids.append(chain.get_id())
        valid_coords = []
        valid_c_alpha_coords = []
        valid_n_coords = []
        valid_c_coords = []
        valid_lengths = []
        invalid_chain_ids = []
        for i, chain in enumerate(rec):
            if chain.get_id() in valid_chain_ids:
                valid_coords.append(coords[i])
                valid_c_alpha_coords.append(c_alpha_coords[i])
                valid_n_coords.append(n_coords[i])
                valid_c_coords.append(c_coords[i])
                valid_lengths.append(lengths[i])
            else:
                invalid_chain_ids.append(chain.get_id())
        # list with n_residues arrays: [n_atoms, 3]
        coords = [item for sublist in valid_coords for item in sublist]

        c_alpha_coords = np.concatenate(
            valid_c_alpha_coords, axis=0)  # [n_residues, 3]
        n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
        c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]

        for invalid_id in invalid_chain_ids:
            rec.detach_child(invalid_id)

        assert len(c_alpha_coords) == len(n_coords)
        assert len(c_alpha_coords) == len(c_coords)
        assert sum(valid_lengths) == len(c_alpha_coords)
        return rec, coords, c_alpha_coords, n_coords, c_coords

    def len(self):
        if self.random_sampling:
            return len(self.slices['x']) - 1
        else:
            return self.nums_amino_cum[-1]

    def get_statistic_info(self):
        node_num = torch.zeros(self.length_total)
        edge_num = torch.zeros(self.length_total)
        for i in tqdm(range(self.length_total)):
            graph = self.get(i)
            node_num[i] = graph.x.shape[0]
            edge_num[i] = graph.edge_index.shape[1]
        num_node_min = torch.min(node_num)
        num_node_max = torch.max(node_num)
        num_node_avg = torch.mean(node_num)
        num_edge_min = torch.min(edge_num)
        num_edge_max = torch.max(edge_num)
        num_edge_avg = torch.mean(edge_num)
        print(f'Graph Num: {self.length_total}')
        print(
            f'Min Nodes: {num_node_min:.2f} Max Nodes: {num_node_max:.2f}. Avg Nodes: {num_node_avg:.2f}')
        print(
            f'Min Edges: {num_edge_min:.2f} Max Edges: {num_edge_max:.2f}. Avg Edges: {num_edge_avg:.2f}')

    def get(self, idx):
        if self.random_sampling:
            idx_protein = idx
        else:
            idx_protein = torch.where(idx < self.nums_amino_cum)[0][0] - 1
        idx_x0 = self.slices['x'][idx_protein]
        idx_x1 = self.slices['x'][idx_protein+1]
        idx_edge0 = self.slices['edge_index'][idx_protein]
        idx_edge1 = self.slices['edge_index'][idx_protein+1]

        if self.diffusion:
            if self.pre_equivariant:
                coors = self.data.pos[idx_x0:idx_x1, :]
                edge_index = edge_index=self.data.edge_index[:, idx_edge0:idx_edge1]
                rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
                rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)
                edge_attr=self.data.edge_attr[idx_edge0:idx_edge1, :]
                edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
            else:
                edge_index = edge_index=self.data.edge_index[:, idx_edge0:idx_edge1]
                edge_attr_feats = self.data.edge_attr[idx_edge0:idx_edge1, :]
            
            extra_x = self.data.x[idx_x0:idx_x1, 20:]
            mu_r_norm=self.data.mu_r_norm[idx_x0:idx_x1, :]
            extra_x_feature = torch.cat([extra_x,mu_r_norm],dim = 1)


            data = Data(
            x=self.data.x[idx_x0:idx_x1, :20],
            extra_x = extra_x_feature,
            pos=self.data.pos[idx_x0:idx_x1, :],
            edge_index=edge_index,
            edge_attr=edge_attr_feats,
            ss = self.data.ss[idx_x0:idx_x1,:],
            y = torch.tensor((), dtype=self.data.x.dtype).unsqueeze(0),
        )
            generate_graph = Data.clone(data)

        else:
            data = Data(
                x=self.data.x[idx_x0:idx_x1, :],
                pos=self.data.pos[idx_x0:idx_x1, :],
                mu_r_norm=self.data.mu_r_norm[idx_x0:idx_x1, :],
                edge_index=self.data.edge_index[:, idx_edge0:idx_edge1],
                edge_attr=self.data.edge_attr[idx_edge0:idx_edge1, :]
            )
            if self.random_sampling:
                remaining_residual = data.x[:, :20].argmax(dim=1).unique()
                select_residual = remaining_residual[torch.multinomial(
                    torch.ones(remaining_residual.shape[0]), 1).item()]
                select_index = (data.x[:, :20].argmax(dim=1)
                                == select_residual).nonzero().view(-1)
                subidx = select_index[torch.multinomial(
                    torch.ones(select_index.shape[0]), 1).item()]
            else:
                subidx = int(idx - self.nums_amino_cum[idx_protein])
            generate_graph = self.generate_one(data, subidx)

            if self.use_micro:
                generate_graph = self.get_calpha_graph_single(
                    generate_graph, subidx)

            
            

        return generate_graph

    def generate_one(self, graph, amino_idx):
        graph2 = Data.clone(graph)
        graph2.x[amino_idx, :] = 0
        graph2.mu_r_norm[amino_idx, :] = 0
        graph2.y = np.argmax(graph.x[amino_idx, :self.num_residue_type])
        return graph2

    def find_idx(self, idx_protein, amino_idx):
        idx = (self.distances[idx_protein][:-1, amino_idx]
               < self.micro_radius).nonzero(as_tuple=True)[0]
        return idx

    def get_calpha_graph_single(self, graph, idx_protein, amino_idx):
        choosen_amino_idx = self.find_idx(idx_protein, amino_idx)
        keep_edge_index = []
        for edge_idx in range(graph.num_edges):
            edge = graph.edge_index.t()[edge_idx]
            if (edge[0] in choosen_amino_idx) and (edge[1] in choosen_amino_idx):
                keep_edge_index.append(edge_idx)
        graph1 = Data(x=graph.x[choosen_amino_idx, :],
                      pos=graph.pos[choosen_amino_idx, :],
                      edge_index=graph.edge_index[:, keep_edge_index],
                      edge_attr=graph.edge_attr[keep_edge_index, :],
                      mu_r_norm=graph.mu_r_norm[choosen_amino_idx, :])
        return graph1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def distance_featurizer(self, dist_list, divisor) -> torch.Tensor:
        # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
        length_scale_list = [1.5 ** x for x in range(15)]
        center_list = [0. for _ in range(15)]

        num_edge = len(dist_list)
        dist_list = np.array(dist_list)

        transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                            for length_scale, center in zip(length_scale_list, center_list)]

        transformed_dist = np.array(transformed_dist).T
        transformed_dist = transformed_dist.reshape((num_edge, -1))
        return torch.from_numpy(transformed_dist.astype(np.float32))


if __name__ == '__main__':
    dataset_arg = dataset_argument(n=51)
    train_dataset = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                              divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                              c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
                              set_length=dataset_arg['set_length'],
                              struc_2nds_res_path = dataset_arg['struc_2nds_res_path'],
                              random_sampling=True,diffusion=True)
    print(len(train_dataset))
    print(train_dataset[10])
