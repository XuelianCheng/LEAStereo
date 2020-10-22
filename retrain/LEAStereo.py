import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.build_model_2d import Disp
from models.decoding_formulas import network_layer_to_space
from retrain.new_model_2d import newFeature
from retrain.skip_model_3d import newMatching

class LEAStereo(nn.Module):
    def __init__(self, args):
        super(LEAStereo, self).__init__()

        network_path_fea, cell_arch_fea = np.load(args.net_arch_fea), np.load(args.cell_arch_fea)
        network_path_mat, cell_arch_mat = np.load(args.net_arch_mat), np.load(args.cell_arch_mat)
        print('Feature network path:{}\nMatching network path:{} \n'.format(network_path_fea, network_path_mat))

        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.maxdisp = args.maxdisp
        self.feature = newFeature(network_arch_fea, cell_arch_fea, args=args)
        self.matching= newMatching(network_arch_mat, cell_arch_mat, args=args) 
        self.disp = Disp(self.maxdisp)

    def forward(self, x, y):
        x = self.feature(x)       
        y = self.feature(y) 

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3]).zero_() 
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        
        cost = self.matching(cost)     
        disp = self.disp(cost)    
        return disp

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params
