import torch.nn as nn
import torch.nn.functional as F
import models.cell_level_search_2d as cell_level_search
from models.genotypes_2d import PRIMITIVES
from models.operations_2d import *
from models.decoding_formulas import Decoder
import pdb

class DispEntropy(nn.Module):
    def __init__(self, maxdisp):
        super(DispEntropy, self).__init__()
        self.softmax = nn.Softmin(dim=1)
        self.maxdisp = maxdisp

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        e = torch.sum(-F.softmax(x,dim=1) * F.log_softmax(x,dim=1),1)
        m = 1.0- torch.isnan(e).type(torch.cuda.FloatTensor)
        x = e*m
        x = self.softmax(x)
        return x

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x


class AutoFeature(nn.Module):
    def __init__(self, num_layers, filter_multiplier=8, block_multiplier=4, step=4, cell=cell_level_search.Cell):
        super(AutoFeature, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._initialize_alphas_betas()
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)
        self._num_end = f_initial * self._block_multiplier

        print('Feature Net block_multiplier:{0}'.format(block_multiplier))
        print('Feature Net filter_multiplier:{0}'.format(filter_multiplier))
        print('Feature Net f_initial:{0}'.format(f_initial))

        self.stem0 = ConvBR(3, half_f_initial * self._block_multiplier, 3, stride=1, padding=1)
        self.stem1 = ConvBR(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, stride=3, padding=1)
        self.stem2 = ConvBR(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=1, padding=1)

        for i in range(self._num_layers):
            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.last_3  = ConvBR(self._num_end , self._num_end, 1, 1, 0, bn=False, relu=False) 
        self.last_6  = ConvBR(self._num_end*2 , self._num_end,    1, 1, 0)  
        self.last_12 = ConvBR(self._num_end*4 , self._num_end*2,  1, 1, 0)  
        self.last_24 = ConvBR(self._num_end*8 , self._num_end*4,  1, 1, 0)  

    def forward(self, x):
        self.level_3 = []
        self.level_6 = []
        self.level_12 = []
        self.level_24 = []

        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)

        self.level_3.append(stem2)

        count = 0
        normalized_betas = torch.randn(self._num_layers, 4, 3).cuda()
        # Softmax on alphas and betas
        if torch.cuda.device_count() > 1:
            #print('more than 1 gpu used!')
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)
            
            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:1].to(device=img_device), dim=-1) * (2/3)

        else:
            normalized_alphas = F.softmax(self.alphas, dim=-1)

            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1) * (2/3)


        for layer in range(self._num_layers):

            if layer == 0:
                level3_new, = self.cells[count](None, None, self.level_3[-1], None, normalized_alphas)
                count += 1
                level6_new, = self.cells[count](None, self.level_3[-1], None, None, normalized_alphas)
                count += 1

                level3_new = normalized_betas[layer][0][1] * level3_new
                level6_new = normalized_betas[layer][0][2] * level6_new
                self.level_3.append(level3_new)
                self.level_6.append(level6_new)

            elif layer == 1:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2 = self.cells[count](None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               None,
                                                               normalized_alphas)
                count += 1
                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][2] * level6_new_2

                level12_new, = self.cells[count](None,
                                                 self.level_6[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level12_new = normalized_betas[layer][1][2] * level12_new
                count += 1

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)

            elif layer == 2:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2, level6_new_3 = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             self.level_12[-1],
                                                                             normalized_alphas)
                count += 1
                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][1] * level6_new_2 + normalized_betas[layer][2][
                    0] * level6_new_3

                level12_new_1, level12_new_2 = self.cells[count](None,
                                                                 self.level_6[-1],
                                                                 self.level_12[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level12_new = normalized_betas[layer][1][2] * level12_new_1 + normalized_betas[layer][2][1] * level12_new_2

                level24_new, = self.cells[count](None,
                                                 self.level_12[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas)
                level24_new = normalized_betas[layer][2][2] * level24_new
                count += 1

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            elif layer == 3:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2, level6_new_3 = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             self.level_12[-1],
                                                                             normalized_alphas)
                count += 1
                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][1] * level6_new_2 + normalized_betas[layer][2][
                    0] * level6_new_3

                level12_new_1, level12_new_2, level12_new_3 = self.cells[count](self.level_12[-2],
                                                                                self.level_6[-1],
                                                                                self.level_12[-1],
                                                                                self.level_24[-1],
                                                                                normalized_alphas)
                count += 1
                level12_new = normalized_betas[layer][1][2] * level12_new_1 + normalized_betas[layer][2][1] * level12_new_2 + normalized_betas[layer][3][
                    0] * level12_new_3

                level24_new_1, level24_new_2 = self.cells[count](None,
                                                                 self.level_12[-1],
                                                                 self.level_24[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level24_new = normalized_betas[layer][2][2] * level24_new_1 + normalized_betas[layer][3][1] * level24_new_2

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            else:
                level3_new_1, level3_new_2 = self.cells[count](self.level_3[-2],
                                                               None,
                                                               self.level_3[-1],
                                                               self.level_6[-1],
                                                               normalized_alphas)
                count += 1
                level3_new = normalized_betas[layer][0][1] * level3_new_1 + normalized_betas[layer][1][0] * level3_new_2

                level6_new_1, level6_new_2, level6_new_3 = self.cells[count](self.level_6[-2],
                                                                             self.level_3[-1],
                                                                             self.level_6[-1],
                                                                             self.level_12[-1],
                                                                             normalized_alphas)
                count += 1

                level6_new = normalized_betas[layer][0][2] * level6_new_1 + normalized_betas[layer][1][1] * level6_new_2 + normalized_betas[layer][2][
                    0] * level6_new_3

                level12_new_1, level12_new_2, level12_new_3 = self.cells[count](self.level_12[-2],
                                                                                self.level_6[-1],
                                                                                self.level_12[-1],
                                                                                self.level_24[-1],
                                                                                normalized_alphas)
                count += 1
                level12_new = normalized_betas[layer][1][2] * level12_new_1 + normalized_betas[layer][2][1] * level12_new_2 + normalized_betas[layer][3][
                    0] * level12_new_3

                level24_new_1, level24_new_2 = self.cells[count](self.level_24[-2],
                                                                 self.level_12[-1],
                                                                 self.level_24[-1],
                                                                 None,
                                                                 normalized_alphas)
                count += 1
                level24_new = normalized_betas[layer][2][2] * level24_new_1 + normalized_betas[layer][3][1] * level24_new_2

                self.level_3.append(level3_new)
                self.level_6.append(level6_new)
                self.level_12.append(level12_new)
                self.level_24.append(level24_new)

            self.level_3 = self.level_3[-2:]
            self.level_6 = self.level_6[-2:]
            self.level_12 = self.level_12[-2:]
            self.level_24 = self.level_24[-2:]

        #define upsampling
        h, w = stem2.size()[2], stem2.size()[3]
        upsample_6  = nn.Upsample(size=stem2.size()[2:], mode='bilinear', align_corners=True)
        upsample_12 = nn.Upsample(size=[h//2, w//2], mode='bilinear', align_corners=True)
        upsample_24 = nn.Upsample(size=[h//4, w//4], mode='bilinear', align_corners=True)

        result_3  = self.last_3(self.level_3[-1])
        result_6  = self.last_3(upsample_6(self.last_6(self.level_6[-1])))
        result_12 = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(self.level_12[-1])))))
        result_24 = self.last_3(upsample_6(self.last_6(upsample_12(self.last_12(self.last_24(self.level_24[-1]))))))      

        sum_feature_map =result_3 + result_6 + result_12 + result_24
        return sum_feature_map

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

