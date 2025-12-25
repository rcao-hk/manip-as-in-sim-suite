CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --encoder vitl --dataset 'HAMMER' --model-path cdm_l515.ckpt --method 'cdm_l515_zs_518x518' --camera 'd435' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/HAMMER_test.txt'

# CUDA_VISIBLE_DEVICES=4 python infer_mixed_dataset.py --encoder vitl --dataset 'HouseCat6D' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/HouseCat6D_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'TransCG' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --dataset_root '/data/robotarm/dataset' --camera 'd435' --split '/home/robotarm/object_depth_percetion/dataset/splits/TransCG_d435_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'XYZ-IBD' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/XYZ-IBD_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'XYZ-IBD' --model-path cdm_l515.ckpt --method 'cdm_l515_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/XYZ-IBD_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'GN-Trans' --model-path cdm_d435.ckpt --method 'cdm_d435_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/GN-Trans_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'XYZ-IBD' --model-path cdm_kinect.ckpt --method 'cdm_kinect_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/XYZ-IBD_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'ROBI' --model-path cdm_kinect.ckpt --method 'cdm_kinect_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/ROBI_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'YCB-V' --model-path cdm_kinect.ckpt --method 'cdm_kinect_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/YCB-V_test.txt'

# CUDA_VISIBLE_DEVICES=0 python infer_mixed_dataset.py --encoder vitl --dataset 'ROBI' --model-path cdm_l515.ckpt --method 'cdm_l515_zs_518x518' --dataset_root '/data/robotarm/dataset' --split '/home/robotarm/object_depth_percetion/dataset/splits/ROBI_test.txt'