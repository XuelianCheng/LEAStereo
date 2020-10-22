import os
import sys
import numpy as np
import torch
from models.decoding_formulas import Decoder
from config_utils.decode_args import obtain_decode_args

class Loader(object):
    def __init__(self, args):
        self.args = args
        # Resuming checkpoint
        self.best_pred = 0.0
        assert args.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(args.resume))
        assert os.path.isfile(args.resume), RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']

        self._alphas_fea = checkpoint['state_dict']['feature.alphas']
        self._betas_fea  = checkpoint['state_dict']['feature.betas']
        self.decoder_fea = Decoder(alphas=self._alphas_fea, betas=self._betas_fea, steps=self.args.step)

        self._alphas_mat = checkpoint['state_dict']['matching.alphas']
        self._betas_mat  = checkpoint['state_dict']['matching.betas']
        self.decoder_mat = Decoder(alphas=self._alphas_mat, betas=self._betas_mat, steps=self.args.step)

    def retreive_alphas_betas(self):
        return self._alphas_fea, self._betas_fea, self._alphas_mat, self._betas_mat

    def decode_architecture(self):
        fea_paths, fea_paths_space = self.decoder_fea.viterbi_decode()
        mat_paths, mat_paths_space = self.decoder_mat.viterbi_decode()
        return fea_paths, fea_paths_space, mat_paths, mat_paths_space

    def decode_cell(self):
        fea_genotype = self.decoder_fea.genotype_decode()
        mat_genotype = self.decoder_mat.genotype_decode()
        return fea_genotype, mat_genotype

def get_new_network_cell():
    args = obtain_decode_args()
    load_model = Loader(args)
    fea_net_paths, fea_net_paths_space, mat_net_paths, mat_net_paths_space = load_model.decode_architecture()
    fea_genotype, mat_genotype = load_model.decode_cell()
    print('Feature Net search results:', fea_net_paths)
    print('Matching Net search results:', mat_net_paths)
    print('Feature Net cell structure:', fea_genotype)
    print('Matching Net cell structure:', mat_genotype)

    dir_name = os.path.dirname(args.resume)
    fea_net_path_filename = os.path.join(dir_name, 'feature_network_path')
    fea_genotype_filename = os.path.join(dir_name, 'feature_genotype')
    np.save(fea_net_path_filename, fea_net_paths)
    np.save(fea_genotype_filename, fea_genotype)

    mat_net_path_filename = os.path.join(dir_name, 'matching_network_path')
    mat_genotype_filename = os.path.join(dir_name, 'matching_genotype')
    np.save(mat_net_path_filename, mat_net_paths)
    np.save(mat_genotype_filename, mat_genotype)

    fea_cell_name = os.path.join(dir_name, 'feature_cell_structure')    
    mat_cell_name = os.path.join(dir_name, 'matching_cell_structure')

if __name__ == '__main__':
    get_new_network_cell()
