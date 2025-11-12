python train_housecat.py --gpus 3 --dataset housecat --mode r --config config/housecat_restored_drnet.yaml --depth_type 'restored' --restored_depth_root '/mnt/DATA/robotarm/rcao/result/depth/HouseCat6D/drnet_zs_448x448'

# python train_housecat.py --gpus 0 --dataset housecat --mode r --config config/housecat_gt.yaml --depth_type 'gt'