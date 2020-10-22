CUDA_VISIBLE_DEVICES=1 python predict.py \
                --kitti2012=1    --maxdisp=192 \
                --crop_height=384  --crop_width=1248  \
                --data_path='./dataset/kitti2012/testing/' \
                --test_list='./dataloaders/lists/kitti2012_test.list' \
                --save_path='./predict/kitti2012/images/' \
                --fea_num_layer 6 --mat_num_layers 12\
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/experiment/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/experiment/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/experiment/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/experiment/matching_genotype.npy' \
                --resume './run/Kitti12/best/best_1.16.pth' 

