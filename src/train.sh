export PYTHONPATH=$PWD:$PYTHONPATH
# python3 diffusion/gradeif.py

# CUDA_VISIBLE_DEVICES=0 python3 diffusion/gradeif.py --lr 5e-4 --wd 1e-5 --drop_out 0.1 --depth 6 --hidden 128 --embedding --embedding_dim 128 --norm_feat --Date Mar15 --noise_type noise


CUDA_VISIBLE_DEVICES=0 python3 diffusion/gradeif.py --lr 5e-4 --wd 1e-5 --drop_out 0.1 --depth 6 --hidden 128 --embedding --embedding_dim 128 --norm_feat --Date Mar15 --noise_type blosum

