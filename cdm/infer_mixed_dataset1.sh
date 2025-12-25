CUDA_VISIBLE_DEVICES=2 python infer_mixed_dataset.py --encoder vitl --dataset 'HAMMER' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --camera 'tof' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/HAMMER_test.txt'

CUDA_VISIBLE_DEVICES=2 python infer_mixed_dataset.py --encoder vitl --dataset 'HAMMER' --model-path cdm_l515.ckpt --method 'cdm_l515_zs_518x518' --camera 'tof' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/HAMMER_test.txt'


# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --encoder vitl --dataset 'YCB-V' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/YCB-V_test.txt'

# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --encoder vitl --dataset 'YCB-V' --model-path cdm_l515.ckpt --method 'cdm_l515_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/YCB-V_test.txt'

# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --encoder vitl --dataset 'T-LESS' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/T-LESS_test_primesense.txt'

# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --encoder vitl --dataset 'T-LESS' --model-path cdm_l515.ckpt --method 'cdm_l515_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/T-LESS_test_primesense.txt'

# CUDA_VISIBLE_DEVICES=1 python infer_mixed_dataset.py --encoder vitl --dataset 'T-LESS' --model-path cdm_kinect.ckpt --method 'cdm_kinect_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/T-LESS_test_primesense.txt'

