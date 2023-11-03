import os
import sys
first_path = sys.path[0]
parent_path = os.path.dirname(first_path)
sys.path.insert(0, parent_path)
from diffusion.model.egnn_pytorch.egnn_pytorch import EGNN, EGNN_Network
from diffusion.model.egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse, EGNN_Sparse_Network
