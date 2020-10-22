import torch
import torch.nn as nn
import numpy as np
from models.genotypes_3d import PRIMITIVES
from models.genotypes_3d import Genotype
from models.operations_3d import *
import torch.nn.functional as F
import numpy as np
import pdb

class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)
        self.preprocess = ConvBR(self.C_prev, self.C_out, 1, 1, 0)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_d = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_h = self.scale_dimension(s1.shape[3], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[4], self.scale)
            s1 = F.interpolate(s1, [feature_size_d, feature_size_h, feature_size_w], mode='trilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]) or (s0.shape[4] != s1.shape[4]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3], s1.shape[4]),
                                            mode='trilinear', align_corners=True)
        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1) 
        return prev_input, concat_feature

class newMatching(nn.Module):
    def __init__(self, network_arch, cell_arch, cell=Cell, args=None):        
        super(newMatching, self).__init__()
        self.args = args
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._step = args.mat_step
        self._num_layers = args.mat_num_layers
        self._block_multiplier = args.mat_block_multiplier
        self._filter_multiplier = args.mat_filter_multiplier
        
        initial_fm = self._filter_multiplier * self._block_multiplier
        half_initial_fm = initial_fm // 2

        self.stem0 = ConvBR(initial_fm*2, initial_fm, 3, stride=1, padding=1)
        self.stem1 = ConvBR(initial_fm, initial_fm, 3, stride=1, padding=1)

        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()
            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, initial_fm / self._block_multiplier,
                             initial_fm / self._block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier * filter_param_dict[level],
                             downup_sample, self.args)

            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier * filter_param_dict[level],
                                 downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier * filter_param_dict[level], downup_sample, self.args)

            self.cells += [_cell]

        self.last_3  = ConvBR(initial_fm, 1, 3, 1, 1,  bn=False, relu=False)  
        self.last_6  = ConvBR(initial_fm*2 , initial_fm,    1, 1, 0)  
        self.last_12 = ConvBR(initial_fm*4 , initial_fm*2,  1, 1, 0)  
        self.last_24 = ConvBR(initial_fm*8 , initial_fm*4,  1, 1, 0)  
        

    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        out = (stem0, stem1)

        for i in range(self._num_layers):
            out = self.cells[i](out[0], out[1])
        last_output = out[-1]

        #define upsampling
        d, h, w = x.size()[2], x.size()[3], x.size()[4]
        upsample_6  = nn.Upsample(size=x.size()[2:], mode='trilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[d//2, h//2, w//2], mode='trilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[d//4, h//4, w//4], mode='trilinear', align_corners=True)

        if last_output.size()[3] == h:
            mat = self.last_3(last_output)
        elif last_output.size()[3] == h//2:
            mat = self.last_3(upsample_6(self.last_6(last_output)))
        elif last_output.size()[3] == h//4:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(last_output)))))
        elif last_output.size()[3] == h//8:
            mat = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(upsample_24(self.last_24(last_output)))))))      
        return mat  

