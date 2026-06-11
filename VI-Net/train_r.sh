# python train_housecat.py --gpus 3 --dataset housecat --mode r --config config/housecat_restored_cdm.yaml --depth_type 'restored' --restored_depth_root '/mnt/DATA/robotarm/rcao/result/depth/HouseCat6D/cdm_d435_zs_518x518'

# python train_housecat.py --gpus 0 --dataset housecat --mode r --config config/housecat_gt.yaml --depth_type 'gt'

python train_housecat.py --gpus 1 --dataset housecat --mode r --config config/housecat_restored_conf.yaml --depth_type 'restored_conf' --conf_thres 0.1 --restored_depth_root '/mnt/DATA/robotarm/rcao/result/depth/HouseCat6D/dreds_clearpose_hiss_50k_dav2_complete_obs_iter_unc_cali_convgru_l1_only_scale_norm_robust_init_wo_soft_fuse_l1+grad_sigma_conf_518x518_seed1/vitl'

python train_housecat.py --gpus 1 --dataset housecat --mode r --config config/housecat_restored_seed0.yaml --depth_type 'restored' --conf_thres 0.1 --restored_depth_root '/mnt/DATA/robotarm/rcao/result/depth/HouseCat6D/dreds_clearpose_hiss_50k_dav2_complete_obs_iter_unc_cali_convgru_l1_only_scale_norm_robust_init_wo_soft_fuse_l1+grad_sigma_conf_518x518_seed0/vitl'

python train_housecat.py --gpus 1 --dataset housecat --mode r --config config/housecat_restored_conf_seed0.yaml --depth_type 'restored_conf' --conf_thres 0.1 --restored_depth_root '/mnt/DATA/robotarm/rcao/result/depth/HouseCat6D/dreds_clearpose_hiss_50k_dav2_complete_obs_iter_unc_cali_convgru_l1_only_scale_norm_robust_init_wo_soft_fuse_l1+grad_sigma_conf_518x518_seed0/vitl'