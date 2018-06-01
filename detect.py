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
with open('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/valid_list.txt') as f:
    valid_list = [line.strip() for line in f.readlines()]
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
dset_val = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, valid_list, mean, std, augment=False, depth_folder=DEPTH_FOLDER, gray_folder=GRAY_FOLDER, surf_folder=SURF_FOLDER)
valid_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)
mt = Metrics()

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
    seg = F.sigmoid(seg).squeeze(0).squeeze(0).detach().cpu().numpy()

    # retrieve detections
    grid_size = 2.0
    dets = grid_size*coords[:, :2] + anchors[0]
    dets_gt = ys[0, :, :2] + anchors[0]
    
    # get input image
    #im_arr = xs.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
    im_path = os.path.join(RGB_FOLDER, valid_list[k]+'.jpg')
    im = Image.open(im_path)
    #im = Image.fromarray(im_arr.astype('uint8'))

    # update metric
    pos_gt_ind = prob_gt > 0
    pos_pred_ind = prob > .5
    dets_gt = dets_gt[pos_gt_ind]
    dets = dets[pos_pred_ind]
    prob_gt = prob_gt[pos_gt_ind]
    prob = prob[pos_pred_ind]

    # apply nms
    dets, prob = nms(dets.detach().cpu().numpy(), prob.detach().cpu().numpy())
    #dets, prob = dets.detach().cpu().numpy(), prob.detach().cpu().numpy()
    # draw outputs
    seg_im = compose_im(np.array(im), seg)
    draw = ImageDraw.Draw(seg_im)
    for p, det in zip(prob, dets):
        x, y = det
        draw.ellipse((x-2, y-2, x+2, y+2), fill='red')

    # draw ground truth - Debug
    for p, det in zip(prob_gt, dets_gt):
        x, y = det
        draw.ellipse((x-1, y-1, x+1, y+1), fill='blue')

    seg_im = seg_im.resize((512, 512))
    mt.forward(valid_list[k], dets_gt.cpu().numpy(), dets)
    seg_im.save('output/' + valid_list[k] + '.jpg')

    # plt.imshow(seg_im)
    # plt.figure()
    # plt.imshow(im)
    # plt.show()

recall, precision = mt.calc_metrics()
print(2.0*precision*recall/(precision+recall+1e-8))
mt.print_metrics()