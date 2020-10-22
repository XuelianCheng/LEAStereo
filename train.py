from __future__ import print_function
import argparse
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import skimage
import pdb
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from collections import OrderedDict
from retrain.LEAStereo import LEAStereo

from mypath import Path
from dataloaders import make_data_loader
from utils.multadds_count import count_parameters_in_MB, comp_multadds, comp_multadds_fw
from config_utils.train_args import obtain_train_args


opt = obtain_train_args()
print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
kwargs = {'num_workers': opt.threads, 'pin_memory': True, 'drop_last':True}
training_data_loader, testing_data_loader = make_data_loader(opt, **kwargs)

print('===> Building model')
model = LEAStereo(opt)

## compute parameters
#print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#print('Number of Feature Net parameters: {}'.format(sum([p.data.nelement() for p in model.feature.parameters()])))
#print('Number of Matching Net parameters: {}'.format(sum([p.data.nelement() for p in model.matching.parameters()])))

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
   
#mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
#print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

torch.backends.cudnn.benchmark = True

if opt.solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999))
elif opt.solver == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def train(epoch):
    epoch_loss = 0
    epoch_error = 0
    valid_iteration = 0
    
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), (batch[2])
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target=torch.squeeze(target,1)
        mask = target < opt.maxdisp
        mask.detach_()
        valid = target[mask].size()[0]
        train_start_time = time()
        if valid > 0:
            model.train()
    
            optimizer.zero_grad()
            disp = model(input1,input2) 
            loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')
            loss.backward()
            optimizer.step()
            
            error = torch.mean(torch.abs(disp[mask] - target[mask])) 
            train_end_time = time()
            train_time = train_end_time - train_start_time

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error += error.item()
            print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), loss.item(), error.item(), train_time))
            sys.stdout.flush()                        
    print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))

def val():
    epoch_error = 0
    valid_iteration = 0
    three_px_acc_all = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()
        target=torch.squeeze(target,1)
        mask = target < opt.maxdisp
        mask.detach_()
        valid=target[mask].size()[0]
        if valid>0:
            with torch.no_grad(): 
                disp = model(input1,input2)
                error = torch.mean(torch.abs(disp[mask] - target[mask])) 

                valid_iteration += 1
                epoch_error += error.item()              
                #computing 3-px error#                
                pred_disp = disp.cpu().detach() 
                true_disp = target.cpu().detach()
                disp_true = true_disp
                index = np.argwhere(true_disp<opt.maxdisp)
                disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
                correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
                three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))

                three_px_acc_all += three_px_acc
    
                print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), three_px_acc))
                sys.stdout.flush()

    print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error/valid_iteration, three_px_acc_all/valid_iteration))
    return three_px_acc_all/valid_iteration

def save_checkpoint(save_path, epoch,state, is_best):
    filename = save_path + "epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + 'best.pth')
    print("Checkpoint saved to {}".format(filename))

if __name__ == '__main__':
    error=100
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        is_best = False
        loss=val()
        if loss < error:
            error=loss
            is_best = True
        if opt.dataset == 'sceneflow':
            if epoch>=0:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
        else:
            if epoch%100 == 0 and epoch >= 3000:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
            if is_best:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)

        scheduler.step()

    save_checkpoint(opt.save_path, opt.nEpochs,{
            'epoch': opt.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
