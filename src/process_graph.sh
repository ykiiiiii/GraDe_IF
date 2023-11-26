export PYTHONPATH=$PWD:$PYTHONPATH
python3 dataset_src/download_pdb.py
python3 dataset_src/generate_graph.py