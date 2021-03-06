import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import glob
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
from dataset.custom_dataloader import ComposedImageData
from torch.utils.data import DataLoader
from dataset.metrics import Metrics
from utils.utils import nms, compose_im
from model.mymodel import MyModel

##############################################################################################################
################################################ Load Model ##################################################
##############################################################################################################

net = torch.load('./best_corner_detector.pth').cuda()
net = net.eval()

RGB_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/rgb'
DEPTH_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/depth'
GRAY_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/gray'
SURF_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/surf'
ANNOT_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/annots'
with open('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/train_list.txt') as f:
    valid_list = [line.strip() for line in f.readlines()]
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
dset_val = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, valid_list, mean, std, augment=False, depth_folder=DEPTH_FOLDER, gray_folder=GRAY_FOLDER, surf_folder=SURF_FOLDER)
valid_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)

##############################################################################################################
############################################# Start Prediction ###############################################
##############################################################################################################
for k, data in enumerate(valid_loader):


    # get the inputs
    xs, ys, es, anchors = data
    xs, ys, es, anchors = Variable(xs.float().cuda()), Variable(ys.float().cuda()), Variable(es.float().cuda()), anchors.float().cuda()

    # run model
    pred, seg = net(xs)
    pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    prob = F.sigmoid(pred[:, 2])
    coords = F.sigmoid(pred[:, :2])
    prob_gt = ys[0, :, 2]
    seg = F.sigmoid(seg).squeeze(0).squeeze(0).detach().cpu().numpy()*255.
    seg_im = Image.fromarray(seg.astype('uint8'))
    seg_im.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/edges/' + valid_list[k] + '.jpg')

