#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from tqdm import tqdm
import itertools
import math

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--normal_loss', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', default='./interior/color/0.png')
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# load image
im = cv2.imread(args.input)
show = im
image_flatten = im.reshape((-1, 3))
depth_unnorm = cv2.imread(args.input.replace('color', 'depth'), -1)
depth = (depth_unnorm - np.min(depth_unnorm)) / (np.max(depth_unnorm) - np.min(depth_unnorm))
depth = depth.reshape((depth.shape[0], depth.shape[1], 1))
data_input = np.concatenate((im/255., depth), 2)
data = torch.from_numpy( np.array([data_input.transpose( (2, 0, 1) ).astype('float32')]) )
#data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')]) )

if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:
    mask = cv2.imread(args.input.replace('.'+args.input.split('.')[-1],'_scribble.png'),0)
    #_, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    #contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('imageshow',mask)
    #cv2.waitKey(0)
    #mask = cv2.drawContours(mask, contours, -1, 8, 5)
    #cv2.imshow('drawimg',mask)
    #cv2.waitKey(0)
    #print(mask.shape)
    mask = mask.reshape(-1)
    print(mask.shape)
    mask_inds = np.unique(mask)
    print(mask_inds)
    mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
    print(mask_inds)
    inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )
    inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )
    target_scr = torch.from_numpy( mask.astype(np.int64) )
    print(inds_sim)
    print(inds_scr)
    print(target_scr)
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable( target_scr )
    # set minLabels
    args.minLabels = len(mask_inds)

# load line_mask
if args.line_mask:
    line = cv2.imread(args.line_path, -1)
    line = line.reshape(-1, 1)
    line_mask = line.repeat(100, axis=1)
##### 20200901 1.转tensor 2.mask 3.取和line_mask做减法最小的值为Loss
fx_d = 5.8262448167737955e+02;
fy_d = 5.8269103270988637e+02;
cx_d = 3.1304475870804731e+02;
cy_d = 2.3844389626620386e+02;

# train
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

# line mask loss definition
loss_line = torch.nn.L2Loss(size_average = True)

# define continuity target
HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)

# define line mask target
im_line_target = torch.zeros(im.shape[0], im.shape[1])

if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
color_avg = np.random.randint(255,size=(100,3))
label_colours = np.random.randint(255,size=(100,3))


for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    outputHP = output.reshape( (im.shape[0], im.shape[1], args.nChannel) )

    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy,HPy_target)
    lhpz = loss_hpz(HPz,HPz_target)

    ignore, target = torch.max( output, 1 )

    line_mask_output = target.copy() # HxWx100
    line_mask_output[line_mask == 0] = 0 # HxWx100

    

    im_target = target.data.cpu().numpy()
    im_target_show = im_target.reshape(im.shape[:2]).astype(np.uint8)

    nLabels = len(np.unique(im_target))

    '''
    # avg color
    if args.visualize:
        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.nChannel:  # update show
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(im.shape)
        cv2.imshow( "output", show )
        cv2.waitKey(10)
    '''
    # random color
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        cv2.imshow( "output", im_target_rgb )
        cv2.waitKey(10)

    # loss 
    if args.scribble:
        loss = args.stepsize_sim * loss_fn(output[ inds_sim ], target[ inds_sim ]) + args.stepsize_scr * loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) + args.stepsize_con * (lhpy + lhpz)
    elif args.normal_loss:
        #print("loss : sim {} | con {} | norm {}".format(args.stepsize_sim * loss_fn(output, target), args.stepsize_con * (lhpy + lhpz), args.stepsize_sim * loss_nm))
        #loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz) + args.stepsize_sim * loss_nm
        loss = args.stepsize_sim * loss_nm
    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
        
    loss.backward()
    optimizer.step()

    print (batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
cv2.imwrite( "output.png", im_target_rgb )
