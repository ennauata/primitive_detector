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
from dataset.custom_dataloader import ComposedImageData
from torch.utils.data import DataLoader
from model.drn import drn_d_105
from model.mymodel import MyModel
from torchsummary import summary
from dataset.metrics import Metrics
from utils.losses import balanced_binary_cross_entropy, mse
from utils.utils import nms

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################

drn = drn_d_105(pretrained=True, channels=8).cuda()
drn = nn.Sequential(*list(drn.children())[:-2]).cuda()
net = MyModel(drn).cuda()

# for params in list(net.parameters())[:-5]:
#     params.requires_grad = False

# for params in net.parameters():
#     print(params.requires_grad)
summary(net, (8, 256, 256))

##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################

# define input folders
RGB_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/rgb'
DEPTH_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/depth'
GRAY_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/gray'
SURF_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/surf'
ANNOT_FOLDER = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/annots'

# define train/val lists
with open('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/train_list.txt') as f:
    train_list = [line.strip() for line in f.readlines()]
with open('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/valid_list.txt') as f:
    valid_list = [line.strip() for line in f.readlines()]

# create dataset
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
dset_train = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, train_list, mean, std, augment=True, depth_folder=DEPTH_FOLDER, gray_folder=GRAY_FOLDER, surf_folder=SURF_FOLDER)
dset_val = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, valid_list, mean, std, augment=False, depth_folder=DEPTH_FOLDER, gray_folder=GRAY_FOLDER, surf_folder=SURF_FOLDER)
dset_list = {'train': dset_train, 'val': dset_val}

# create loaders
train_loader = DataLoader(dset_train, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
dset_loader = {'train': train_loader, 'val': valid_loader}

# select optimizer

optimizer = optim.Adam(filter(lambda x:x.requires_grad, net.parameters()), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
best_score = 0.0
mt = Metrics()

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

for epoch in range(1000):

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            net.train()  # Set model to training mode
        else:
            net.eval()

        running_loss = 0.0
        for i, data in enumerate(dset_loader[phase]):
            # get the inputs
            xs, ys, es, anchors = data
            xs, ys, es, anchors = Variable(xs.float().cuda()), Variable(ys.float().cuda()), Variable(es.float().cuda()), anchors.float().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if phase == 'train':
                pred, seg = net(xs)
            else:
                with torch.no_grad():
                    pred, seg = net(xs)
            pred = pred.permute(0, 2, 3, 1).contiguous().view(pred.shape[0], -1, 3)    
            prob = F.sigmoid(pred[:, :, 2])
            coords = F.sigmoid(pred[:, :, :2])
            seg = F.sigmoid(seg.squeeze(1))

            # compute losses
            l_conf = balanced_binary_cross_entropy(prob, ys[:, :, 2])
            l_loc = mse(coords, ys[:, :, :2], ys[:, :, 2])
            l_seg = F.binary_cross_entropy(seg, es, size_average=True)
            loss = l_conf + l_loc + l_seg
            running_loss += loss

            # step
            if phase == 'train':
                loss.backward()
                optimizer.step()
            # compute precison/recall
            else:

                # retrieve detections
                grid_size = 2.0
                dets = grid_size*coords[0, :, :2] + anchors[0]
                dets_gt = grid_size*ys[0, :, :2] + anchors[0]

                # select detections
                prob_gt = ys[0, :, 2]
                prob = prob.view(-1)
                pos_gt_ind = prob_gt > 0
                pos_pred_ind = prob > .5
                dets_gt = dets_gt[pos_gt_ind]
                dets = dets[pos_pred_ind]
                prob = prob[pos_pred_ind]
                dets, prob = nms(np.array(dets), np.array(prob))

                # update metric
                mt.forward(valid_list[i], np.array(dets_gt), np.array(dets))

        # print epoch loss
        print('[%d] %s lr: %f \nloss: %.5f' %
              (epoch + 1, phase, optimizer.param_groups[0]['lr'], running_loss / len(dset_loader[phase])))

        # tack best model
        if phase == 'val':
            
            recall, precision = mt.calc_metrics()
            f_score = 2.0*precision*recall/(precision+recall+1e-8)
            print('val f_score %.5f' % f_score)
            mt.reset()

            if f_score > best_score:
                print('new best: f_score %.5f' % f_score)
                best_score = f_score
                torch.save(net, './best_corner_detector.pth')

        # reset running loss
        running_loss = 0.0

print('Finished Training')