import os
import json
from tqdm import tqdm
def get_pdb(pdb_code=""):

    os.system(f"wget -qnc -P all/ https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"all/{pdb_code}.pdb"

from Bio.PDB import PDBParser, PDBIO,Select


with open('chain_set_splits.json', 'r') as f:
  data = json.load(f)


exits_file = os.listdir('all/')
for key in data.keys():
    for pdb_code in data[key]:
        pdb_code = pdb_code[:4]
        if pdb_code+'.pdb' in exits_file:
            print(pdb_code,'exist')
        else:
            get_pdb(pdb_code)
            print(pdb_code)


err_file = []
all_processed_file = os.listdir('test/') + os.listdir('train/')+os.listdir('validation/')
for key in data.keys():
    if key not in ['cath_nodes']:
        for pdb_code in tqdm(data[key]):
            if pdb_code+'.pdb' not in all_processed_file:
                pdb_file = f'all/{pdb_code[:4]}'+'.pdb'
                chain_id = pdb_code[5]

                parser = PDBParser(QUIET=True)
                try:
                    structure = parser.get_structure("name", pdb_file)

                    io = PDBIO()

                    class ChainSelector(Select):
                        def accept_chain(self, chain):
                            return chain.get_id() == chain_id

                        def accept_residue(self, residue):
                            return residue.id[0] == " " 
                    io.set_structure(structure)
                    io.save(key+f"/{pdb_code[:4]}.{chain_id}.pdb", ChainSelector())
                except FileNotFoundError:
                    err_file.append(pdb_code)


print(err_file)