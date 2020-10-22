import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np
import pdb


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

    def tensor2array(self, tensor, max_value=255, colormap='rainbow'):
        if max_value is None:
            max_value = tensor.max().numpy()
        if tensor.ndimension() == 2:
            try:
                import cv2
                if cv2.__version__.startswith('2'):# 2.4
                    color_cvt = cv2.cv.CV_BGR2RGB
                else:
                    color_cvt = cv2.COLOR_BGR2RGB
                if colormap == 'rainbow':
                    colormap = cv2.COLORMAP_RAINBOW
                elif colormap == 'bone':
                    colormap = cv2.COLORMAP_BONE
                array = (255*tensor.numpy()/max_value).clip(0, 255).astype(np.uint8)
                colored_array = cv2.applyColorMap(array, colormap)
                array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
                array = array.transpose(2,0,1)
            except ImportError:
                if tensor.ndimension() == 2:
                    tensor.unsqueeze_(2)
                array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

        elif tensor.ndimension() == 3:
            #assert(tensor.size(0) == 3)
            array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5
        return array

    def visualize_image_stereo(self, writer, image, target, output, global_step):
        pr_image = self.tensor2array(output[0].cpu().data, max_value=144, colormap='bone')
        writer.add_image('Predicted disparity', pr_image, global_step)
        gt_image = self.tensor2array(target[0].cpu().data, max_value=144, colormap='bone')
        writer.add_image('Groundtruth disparity', gt_image, global_step)
