python train_housecat.py --gpus 3 --dataset housecat --mode r --config config/housecat_restored_d3roma.yaml --depth_type 'restored' --restored_depth_root '/mnt/DATA/robotarm/rcao/result/depth/HouseCat6D/d3roma_zs_360x640'

# python train_housecat.py --gpus 0 --dataset housecat --mode r --config config/housecat_gt.yaml --depth_type 'gt'