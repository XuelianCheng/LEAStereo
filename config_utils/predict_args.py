import argparse

def obtain_predict_args():

    parser = argparse.ArgumentParser(description='LEStereo Prediction')
    parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--maxdisp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='', help="resume from saved model")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--sceneflow', type=int, default=0, help='sceneflow dataset? Default=False')
    parser.add_argument('--kitti2012', type=int, default=0, help='kitti 2012? Default=False')
    parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    parser.add_argument('--middlebury', type=int, default=0, help='Middlebury? Default=False')
    parser.add_argument('--data_path', type=str, required=True, help="data root")
    parser.add_argument('--test_list', type=str, required=True, help="training list")
    parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
    ######### LEStereo params####################
    parser.add_argument('--fea_num_layers', type=int, default=6)
    parser.add_argument('--mat_num_layers', type=int, default=12)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--mat_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=4)
    parser.add_argument('--mat_block_multiplier', type=int, default=4)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--mat_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default=None, type=str)
    parser.add_argument('--cell_arch_fea', default=None, type=str)
    parser.add_argument('--net_arch_mat', default=None, type=str)
    parser.add_argument('--cell_arch_mat', default=None, type=str)

    args = parser.parse_args()
    return args
