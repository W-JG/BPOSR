

# single-GPUs
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python -W ignore basicsr/train.py -opt options/BPOSR_train/train_BPOSR_SW_SRx4.yml
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=1 python -W ignore basicsr/train.py -opt options/BPOSR_train/train_BPOSR_SW_SRx8.yml
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=2 python -W ignore basicsr/train.py -opt options/BPOSR_train/train_BPOSR_SW_SRx16.yml

# Multi-GPUs
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 \\
python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 --use_env  \\
basicsr/train.py -opt  options/BPOSR_train/train_MPSR_SW_SRx4_light.yml --launcher pytorch \\