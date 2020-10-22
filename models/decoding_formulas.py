import numpy as np
import pdb
import torch
import torch.nn.functional as F

def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    """
        return:
        network_space[layer][level][sample]:
        layer: 0 - 12
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    """
    return space

class Decoder(object):
    def __init__(self, alphas, betas, steps):
        self._betas = betas
        self._alphas = alphas
        self._steps = steps
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(self._num_layers, 4, 3)

        for layer in range(self._num_layers):
            if layer == 0:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)  * (2/3)
            elif layer == 1:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)

            elif layer == 2:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)            
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)


            else:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)
                self.network_space[layer][3][:2] = F.softmax(self._betas[layer][3][:2], dim=-1) * (2/3)
        
    def viterbi_decode(self):

        prob_space = np.zeros((self.network_space.shape[:2]))
        path_space = np.zeros((self.network_space.shape[:2])).astype('int8')

        for layer in range(self.network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = self.network_space[layer][0][1]
                prob_space[layer][1] = self.network_space[layer][0][2]
                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(self.network_space.shape[1]):
                    if layer - sample < - 1:
                        continue
                    local_prob = []
                    for rate in range(self.network_space.shape[2]):  # k[0 : ➚, 1: ➙, 2 : ➘]
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            continue
                        else:
                            local_prob.append(prob_space[layer - 1][sample + 1 - rate] *
                                              self.network_space[layer][sample + 1 - rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path  # path[1 : ➚, 0: ➙, -1 : ➘]

        output_sample = prob_space[-1, :].argmax(axis=-1)
        actual_path = np.zeros(self._num_layers).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self._num_layers):
            actual_path[-i - 1] = actual_path[-i] + path_space[self._num_layers - i, actual_path[-i]]
        return actual_path, network_layer_to_space(actual_path)

    def genotype_decode(self):
        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_alphas = F.softmax(self._alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(normalized_alphas, self._steps)
        return gene_cell
