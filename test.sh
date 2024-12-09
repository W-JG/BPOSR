#CUDA_VISIBLE_DEVICES=0  python basicsr/test.py -opt ./options/BPOSR_test/test_BPOSR_x4.yml 
CUDA_VISIBLE_DEVICES=1 python basicsr/test.py -opt ./options/BPOSR_test/test_BPOSR_x8.yml 
CUDA_VISIBLE_DEVICES=2 python basicsr/test.py -opt ./options/BPOSR_test/test_BPOSR_x16.yml 

