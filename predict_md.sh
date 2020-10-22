CUDA_VISIBLE_DEVICES=0 python predict.py \
                --middlebury=1    --maxdisp=408 \
                --crop_height=1008  --crop_width=1512  \
                --data_path='./dataset/MiddEval3/testH/' \
                --test_list='./dataloaders/lists/middeval3_test.list' \
                --save_path='./predict/middlebury/images/' \
                --fea_num_layer 6 --mat_num_layers 12\
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/architecture/matching_genotype.npy' \
                --resume './run/MiddEval3/best.pth' 

