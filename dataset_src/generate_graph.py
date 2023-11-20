import os
from dataset_src.cath_imem_2nd import Cath_imem,dataset_argument
from torch.optim import Adam
from torch_geometric.data import Batch,Data
from dataset_src.utils import NormalizeProtein
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch.nn.functional as F
import torch
from tqdm import tqdm

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def get_struc2ndRes(pdb_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))

    p = PDBParser()
    structure = p.get_structure('random_id', pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename, dssp='mkdssp')

    # From model, extract the list of amino acids
    model_residues = [(chain.id, residue.id[1]) for chain in model for residue in chain if residue.id[0] == ' ']
    # From DSSP, extract the list of amino acids
    dssp_residues = [(k[0], k[1][1]) for k in dssp.keys()]

    # Determine the missing amino acids
    missing_residues = set(model_residues) - set(dssp_residues)

    # Initialize a list of integers for known secondary structures,
    # and another list of zeroes for one-hot encoding
    integer_encoded = []
    one_hot_list = torch.zeros(len(model_residues), len(struc_2nds_res_alphabet))

    current_position = 0
    for chain_id, residue_num in model_residues:
        dssp_key = (chain_id, (' ', residue_num, ' '))
        if (chain_id, residue_num) not in missing_residues and dssp_key in dssp:
            
            sec_structure_char = dssp[dssp_key][2]
            sec_structure_char = sec_structure_char.replace('-', 'L')
            integer_encoded.append(char_to_int[sec_structure_char])

            one_hot = F.one_hot(torch.tensor(integer_encoded[-1]), num_classes=8)
            one_hot_list[current_position] = one_hot
        else:
            print(pdb_filename,'Missing residue: ', chain_id, residue_num, 'fill with 0')
        current_position += 1
    ss_encoding = one_hot_list[:current_position]
    return ss_encoding

def prepare_graph(data):
    del data['distances']
    del data['edge_dist']
    mu_r_norm=data.mu_r_norm

    extra_x_feature = torch.cat([data.x[:,20:],mu_r_norm],dim=1)
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

def pdb2graph(filename,normalize_path = 'dataset_src/mean_attr.pt'):
    #### dataset  ####
    dataset_arg = dataset_argument(n=51)
    dataset = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                                divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                                c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
                                set_length=dataset_arg['set_length'],
                                struc_2nds_res_path = dataset_arg['struc_2nds_res_path'],
                                random_sampling=True,diffusion=True)
    rec, rec_coords, c_alpha_coords, n_coords, c_coords = dataset.get_receptor_inference(filename)
    struc_2nd_res = get_struc2ndRes(filename)
    rec_graph = dataset.get_calpha_graph(
                rec, c_alpha_coords, n_coords, c_coords, rec_coords, struc_2nd_res)
    if rec_graph:
        normalize_transform = NormalizeProtein(filename=normalize_path)
        
        graph = normalize_transform(rec_graph)
        return graph
    else:
        return None



if __name__ == '__main__':

    # generate a batch of protein graph
    error_pdb = []
    for key in ['test','validation','train']:
        pdb_dir =f'dataset/raw/{key}/'
        save_dir = f'dataset/full_aa/process/{key}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename_list = [i for i in os.listdir(pdb_dir) if i.endswith('.pdb')]
        for filename in tqdm(filename_list):
            if os.path.exists(save_dir+filename.replace('.pdb','.pt')):
                pass
            else:
                try:
                    graph = pdb2graph(pdb_dir+filename,'dataset_src/mean_attr.pt')
                    if graph:
                        torch.save(graph,save_dir+filename.replace('.pdb','.pt'))
                    else:
                        error_pdb.append(filename)
                except (IndexError, KeyError):
                    error_pdb.append(filename)

    print(len(error_pdb))
    print(error_pdb)