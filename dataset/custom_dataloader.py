import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle as p
import glob
from random import randint

class ComposedImageData(Dataset):

    def __init__(self, rgb_folder, annot_folder, id_list, mean, std, augment=False, depth_folder=None, gray_folder=None, surf_folder=None):
        self._data_refs = id_list
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.annot_folder = annot_folder
        self.gray_folder = gray_folder
        self.surf_folder = surf_folder
        self.augment = augment
        self.mean = mean
        self.std = std

    def __getitem__(self, index):

        # load image + annots
        im_path = os.path.join(self.rgb_folder, self._data_refs[index]+'.jpg')
        im = Image.open(im_path)

        if self.depth_folder is not None:
            dp_path = os.path.join(self.depth_folder, self._data_refs[index]+'.jpg')
            dp_im = Image.open(dp_path).convert('L')

        if self.gray_folder is not None:
            gr_path = os.path.join(self.gray_folder, self._data_refs[index]+'.jpg')
            gr_im = Image.open(gr_path).convert('L')

        if self.surf_folder is not None:
            sf_path = os.path.join(self.surf_folder, self._data_refs[index]+'.jpg')
            sf_im = Image.open(sf_path).convert('RGB')

        corners_annot, edges_annot = self.get_annots(index)
        im_ed = self.calc_edge_gt(edges_annot)
        if self.augment:
            rot = randint(0, 3)*90.0
            flip = randint(0, 1) == 1

            # rotate and flip image + corners
            corners_annot = [self.flip_and_rotate(v, flip, rot) for v in corners_annot]
            im = im.rotate(rot)
            im_ed = im_ed.rotate(rot)
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                im_ed = im_ed.transpose(Image.FLIP_LEFT_RIGHT)

            # add depth
            if self.depth_folder is not None:
                dp_im = dp_im.rotate(rot)
                if flip:
                    dp_im = dp_im.transpose(Image.FLIP_LEFT_RIGHT)

            # add gray
            if self.gray_folder is not None:
                gr_im = gr_im.rotate(rot)
                if flip:
                    gr_im = gr_im.transpose(Image.FLIP_LEFT_RIGHT)

            # add surf
            if self.surf_folder is not None:
                sf_im = sf_im.rotate(rot)
                if flip:
                    sf_im = sf_im.transpose(Image.FLIP_LEFT_RIGHT)

        # calculate anchors + ground truth
        anchors = self.generate_anchors()
        gt = self.calc_gt(corners_annot, anchors)

        # convert to numpy array
        im = np.array(im).transpose((2, 0, 1))/255.0
        im = (im-np.array(self.mean)[:, np.newaxis, np.newaxis])/np.array(self.std)[:, np.newaxis, np.newaxis]
        ed = np.array(im_ed)/255.

        if self.depth_folder is not None:
            dp_im = np.array(dp_im)/255.0
            im = np.concatenate([im, dp_im[np.newaxis, :, :]], axis=0)

        if self.gray_folder is not None:
            gr_im = np.array(gr_im)/255.0
            im = np.concatenate([im, gr_im[np.newaxis, :, :]], axis=0)

        if self.surf_folder is not None:
            sf_im = np.array(sf_im).transpose((2, 0, 1))/255.0
            im = np.concatenate([im, sf_im], axis=0)

        # convert to tensor
        im = torch.from_numpy(im)
        gt = torch.from_numpy(gt)
        ed = torch.from_numpy(ed)
        return im, gt, ed, anchors

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self._data_refs)

    # rotate coords
    def rotate(self, image_shape, xy, angle):
        org_center = (image_shape-1)/2.
        rot_center = (image_shape-1)/2.
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        new = new+rot_center
        return new

    def generate_anchors(self, feat_shape=128., grid_size=2.):
        
        # enumerate shifts in feature space
        shifts_y = np.arange(feat_shape) * grid_size
        shifts_x = np.arange(feat_shape) * grid_size
        xx, yy = np.meshgrid(shifts_x, shifts_y)
        anchors = np.stack((xx, yy), axis=2)
        return anchors.reshape((-1, 2))

    def get_annots(self, index):

        # load ground-truth
        gt_path = os.path.join(self.annot_folder, self._data_refs[index]+'.npy')
        v_set = np.load(open(gt_path, "rb"),  encoding='bytes')
        v_set = dict(v_set[()])
        return v_set.keys(), v_set

    def calc_edge_gt(self, annot, shape=256):
        im = Image.fromarray(np.zeros((shape, shape)))
        draw = ImageDraw.Draw(im)
        for c in annot:
            for n in annot[c]:
                draw.line((c[0], c[1], n[0], n[1]), fill='white', width=2)
        return im

    def calc_gt(self, annot, anchors, grid_size=2.0):
        gt = np.zeros((anchors.shape[0], 3))
        for k, anc in enumerate(anchors):
            for ann in annot:
                if (anc[0] <= ann[0] < anc[0]+grid_size) and (anc[1] <= ann[1] < anc[1]+grid_size):
                    gt[k, :2] = (ann[:2]-anc[:2])/grid_size
                    gt[k, 2] = 1.0
        return gt

    def flip_and_rotate(self, v, flip, rot, shape=256.):
        v = self.rotate(np.array((shape, shape)), v, rot)
        if flip:
            x, y = v
            v = (shape/2-abs(shape/2-x), y) if x > shape/2 else (shape/2+abs(shape/2-x), y)
        return v




