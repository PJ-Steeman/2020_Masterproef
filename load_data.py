import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from darknet import Darknet

from median_pool import MedianPool2d

print('starting test read')
im = Image.open('data/horse.jpg').convert('RGB')
print('img read!')


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        confs_if_object = output_objectness #confs_for_class * output_objectness
        confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PatchTransformerKeypoints(nn.Module):
    def __init__(self):
        super(PatchTransformerKeypoints, self).__init__()
        self.medianpooler = MedianPool2d(7,same=True)

    def forward(self, adv_patch, lab_batch, img_size):
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        adv_patch = adv_patch.unsqueeze(0)
        patch_size = adv_patch.size(3)
        # Maak een tensor de grootte van de lab_batch size, hierin komen de patches
        adv_batch = torch.cuda.FloatTensor(lab_batch.size(0), lab_batch.size(1), 3, img_size, img_size).fill_(0)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Deze transformatie voegt een random kleur verandering, rotatie en shaling toe
        random_changes = transforms.Compose([
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.RandomAffine(5, translate=None, scale=(0.95, 1.15), shear=None, resample=2, fillcolor=0)
        ])

        for i in range(batch_size[0]):
            for j in range(batch_size[1]):
                # Haal de keypoints van dit object op
                kp = lab_batch[i,j,:]
                # Indien het torso in beeld is
                if(kp[17] > 0 and kp[20] > 0):
                    # Moest de onderkant van het torso uit beeld zijn schatten we waar dit zou zijn
                    if (kp[38] == 0):
                        kp[37] = torch.abs(kp[18] - kp[15])*1.7+1
                        if(kp[37] > img_size):
                            kp[37] = img_size
                    # We zetten de adv_patch om in een image
                    patch_im = transforms.ToPILImage()(adv_patch[0,0,:].cpu())
                    # We snijden het stuk voorbestemd voor de patch uit de image
                    patch_im = transforms.functional.crop(patch_im, 0, int(1/6*patch_size), int(2/3*patch_size), patch_size)
                    # De patch wordt geresized naar de grootte van het torso
                    patch_im = transforms.functional.resize(patch_im, (int(torch.abs(kp[37] - kp[16])*1.2)+1, int(torch.abs(kp[18] - kp[15])*1.3)+1), interpolation=2)

                    # Bepaal de nodige padding
                    left = min(kp[15], kp[18])
                    right = max(kp[15], kp[18])
                    top = min(kp[16], kp[37])
                    bottom = max(kp[16], kp[37])

                    patch_im = transforms.functional.pad(patch_im, (left, top, img_size - right, img_size - bottom), fill=0, padding_mode='constant')
                    patch_im = random_changes(patch_im)
                    patch_im = transforms.functional.center_crop(patch_im, img_size)

                    # Voeg de random vervormingen toe en zet de patch terug om naar een tensor
                    to_tens = transforms.ToTensor()
                    patch_tens = to_tens(patch_im)
                    adv_batch[i,j,:] = patch_tens

                    # Bepaal of de bovenarmen aangeduid zijn
                    if(kp[17] > 0 and kp[23] > 0 and kp[20] > 0 and kp[26] > 0):
                        # Snij het correcte stuk van de patch uit
                        patch_im_r = transforms.functional.crop(patch_im, 0, int(5/6*patch_size), int(1/2*patch_size), int(1/6*patch_size))
                        patch_im_l = transforms.functional.crop(patch_im, 0, 0, int(1/2*patch_size), int(1/6*patch_size))

                        # Bepaal de arm breedte en de lengte van de bovenarmen
                        arm_w = torch.abs(kp[15] - kp[18])/2.5+1
                        r_len = torch.sqrt((kp[18] - kp[24])**2 + (kp[19] - kp[25])**2)+1
                        l_len = torch.sqrt((kp[15] - kp[21])**2 + (kp[16] - kp[22])**2)+1

                        # Herschaal de patch met deze afstanden
                        patch_im_r = transforms.functional.resize(patch_im, (int(r_len), int(arm_w)), interpolation=2)

                        # Bepaal de hoek waaronder de arm zich bevindt
                        angle = torch.tan((kp[18] - kp[24])/(kp[19] - kp[25]))*57.2958
                        if torch.isnan(angle):
                            angle = 90
                        # Roteer de arm
                        patch_im_r = transforms.functional.rotate(patch_im_r, angle, resample=2, expand=True, center=None)
                        # Doe hetzelfde voor de linker arm
                        patch_im_l = transforms.functional.resize(patch_im, (int(l_len), int(arm_w)), interpolation=2)
                        angle = torch.tan((kp[15] - kp[21])/(kp[16] - kp[22]))*57.2958
                        if torch.isnan(angle):
                            angle = 90
                        patch_im_l = transforms.functional.rotate(patch_im_l, angle, resample=2, expand=True, center=None)

                        # Bepaal de padding
                        left = min(kp[18], kp[24])
                        right = max(kp[18], kp[24])
                        top = min(kp[19], kp[25])
                        bottom = max(kp[19], kp[25])
                        patch_im_r = transforms.functional.pad(patch_im_r, (left, top, img_size - right, img_size - bottom), fill=0, padding_mode='constant')

                        left = min(kp[15], kp[21])
                        right = max(kp[15], kp[21])
                        top = min(kp[16], kp[22])
                        bottom = max(kp[16], kp[22])
                        patch_im_l = transforms.functional.pad(patch_im_l, (left, top, img_size - right, img_size - bottom), fill=0, padding_mode='constant')

                        patch_im_r = random_changes(patch_im_r)
                        patch_im_l = random_changes(patch_im_l)

                        # Door de rotatie kan de patch size een beetje veranderen, we snijden hem duss terug op de correcte lengte
                        patch_im_r = transforms.functional.center_crop(patch_im_r, img_size)
                        patch_im_l = transforms.functional.center_crop(patch_im_l, img_size)

                        patch_tens_r = to_tens(patch_im_r).cuda()
                        patch_tens_l = to_tens(patch_im_l).cuda()

                        # Waar er nog geen patch staat komt onze nieuwe patch te staan
                        adv_batch[i,j,:] = torch.where((adv_batch[i,j,:] == 0), patch_tens_r, adv_batch[i,j,:])
                        adv_batch[i,j,:] = torch.where((adv_batch[i,j,:] == 0), patch_tens_l, adv_batch[i,j,:])

                    # hetzelfde wordt nog is herhaald maar nu voor de onderarm
                    if(kp[29] > 0 and kp[32] > 0 and kp[23] > 0 and kp[26] > 0):
                        patch_im_r = transforms.functional.crop(patch_im, int(1/2*patch_size), int(5/6*patch_size), int(1/2*patch_size), int(1/6*patch_size))
                        patch_im_l = transforms.functional.crop(patch_im, int(1/2*patch_size), 0, int(1/2*patch_size), int(1/6*patch_size))

                        arm_w = torch.abs(kp[15] - kp[18])/2.5+1
                        r_len = torch.sqrt((kp[30] - kp[24])**2 + (kp[31] - kp[25])**2)+1
                        l_len = torch.sqrt((kp[27] - kp[21])**2 + (kp[28] - kp[22])**2)+1

                        patch_im_r = transforms.functional.resize(patch_im, (int(r_len), int(arm_w)), interpolation=2)
                        angle = torch.tan((kp[24] - kp[30])/(kp[25] - kp[31]))*57.2958
                        if torch.isnan(angle):
                            angle = 90
                        patch_im_r = transforms.functional.rotate(patch_im_r, angle, resample=2, expand=True, center=None)

                        patch_im_l = transforms.functional.resize(patch_im, (int(l_len), int(arm_w)), interpolation=2)
                        angle = torch.tan((kp[21] - kp[27])/(kp[22] - kp[28]))*57.2958
                        if torch.isnan(angle):
                            angle = 90
                        patch_im_l = transforms.functional.rotate(patch_im_l, angle, resample=2, expand=True, center=None)

                        left = min(kp[24], kp[30])
                        right = max(kp[24], kp[30])
                        top = min(kp[25], kp[31])
                        bottom = max(kp[25], kp[31])

                        patch_im_r = transforms.functional.pad(patch_im_r, (left, top, img_size - right, img_size - bottom), fill=0, padding_mode='constant')

                        left = min(kp[21], kp[27])
                        right = max(kp[21], kp[27])
                        top = min(kp[22], kp[28])
                        bottom = max(kp[22], kp[28])

                        patch_im_l = transforms.functional.pad(patch_im_l, (left, top, img_size - right, img_size - bottom), fill=0, padding_mode='constant')

                        patch_im_r = random_changes(patch_im_r)
                        patch_im_l = random_changes(patch_im_l)

                        patch_im_r = transforms.functional.center_crop(patch_im_r, img_size)
                        patch_im_l = transforms.functional.center_crop(patch_im_l, img_size)

                        patch_tens_r = to_tens(patch_im_r).cuda()
                        patch_tens_l = to_tens(patch_im_l).cuda()

                        adv_batch[i,j,:] = torch.where((adv_batch[i,j,:] == 0), patch_tens_r, adv_batch[i,j,:])
                        adv_batch[i,j,:] = torch.where((adv_batch[i,j,:] == 0), patch_tens_l, adv_batch[i,j,:])


        return adv_batch

class PatchTransformerKeypointsTensors(nn.Module):
    def __init__(self):
        super(PatchTransformerKeypointsTensors, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.medianpooler = MedianPool2d(7,same=True)
        self.patch_scale = 1

    def forward(self, adv_patch, lab_batch, img_size):

        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        adv_patch = adv_patch.unsqueeze(0)
        patch_size = adv_patch.size(3)

        # Maak een tensor de grootte van de lab_batch size, hierin komen de patches
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Random contrast
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Random helderheid
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Random ruis
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Als er niks gedetecteerd wordt -> masker vol nullen
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Torso patch uitknippen
        adv_patch_torso = torch.narrow(adv_batch, 4, int(patch_size/5), int(3*patch_size/5))
        adv_patch_torso = F.interpolate(adv_patch_torso, size=(3, img_size, img_size))

        # Linker bovenarm uitkinppen
        adv_patch_boven_arm_l = torch.narrow(adv_batch, 4, 0, int(patch_size/5))
        adv_patch_boven_arm_l = torch.narrow(adv_patch_boven_arm_l, 3, 0, int(patch_size/2))
        adv_patch_boven_arm_l = F.interpolate(adv_patch_boven_arm_l, size=(3, img_size, img_size))

        # Rechter bovernarm uitknippen
        adv_patch_boven_arm_r = torch.narrow(adv_batch, 4, int(4*patch_size/5), int(patch_size/5))
        adv_patch_boven_arm_r = torch.narrow(adv_patch_boven_arm_r, 3, 0, int(patch_size/2))
        adv_patch_boven_arm_r = F.interpolate(adv_patch_boven_arm_r, size=(3, img_size, img_size))

        # Linker onderarm uitknippen
        adv_patch_onder_arm_l = torch.narrow(adv_batch, 4, 0, int(patch_size/5))
        adv_patch_onder_arm_l = torch.narrow(adv_patch_onder_arm_l, 3, int(patch_size/2), int(patch_size/2))
        adv_patch_onder_arm_l = F.interpolate(adv_patch_onder_arm_l, size=(3, img_size, img_size))

        # Rechter onderarm uitknippen
        adv_patch_onder_arm_r = torch.narrow(adv_batch, 4, int(4*patch_size/5), int(patch_size/5))
        adv_patch_onder_arm_r = torch.narrow(adv_patch_onder_arm_r, 3, int(patch_size/2), int(patch_size/2))
        adv_patch_onder_arm_r = F.interpolate(adv_patch_onder_arm_r, size=(3, img_size, img_size))

        msk_batch = F.interpolate(msk_batch, size=(3, img_size, img_size))

        current_patch_size = adv_patch.size(-1)

        # Torso gedeelte
        # Als de onderkant niet zichtbaar is schatten waar deze zou zijn
        lab_batch[:, :, 20] = torch.where(lab_batch[:, :, 20] == 0, torch.abs(lab_batch[:, :, 19] - lab_batch[:, :, 16])*1.7+1, lab_batch[:, :, 20])

        # Torso breedte
        target_size_x = torch.abs(lab_batch[:, :, 16] - lab_batch[:, :, 19])+1
        target_size_x *= self.patch_scale

        # Torse lengte
        target_size_y = torch.abs(lab_batch[:, :, 17] - lab_batch[:, :, 35])+1
        target_size_y *= self.patch_scale*0.85

        # Bij mensen in zijaanzicht de patch iets breeder plaatsen
        target_size_x = torch.where(target_size_y > 2*target_size_x, target_size_y/2, target_size_x)

        scale_x = target_size_x / current_patch_size
        scale_y = target_size_y / current_patch_size

        # Plaats locaties
        target_x = ((torch.min(lab_batch[:, :, 16], lab_batch[:, :, 19])+target_size_x*1.15/2)/img_size).view(np.prod(batch_size))
        target_y = ((torch.min(lab_batch[:, :, 17], lab_batch[:, :, 20])+target_size_y*1.15/2)/img_size).view(np.prod(batch_size))

        s = adv_patch_torso.size()
        adv_patch_torso = adv_patch_torso.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        # img = msk_batch.view(s[0], s[1], s[2], s[3], s[4])[0, 0, :, :, :].detach().cpu()
        # im = transforms.ToPILImage('RGB')(img)
        # plt.figure(200)
        # plt.imshow(im)
        # plt.show()
        # exit()

        anglesize = (lab_batch.size(0) * lab_batch.size(1))

        # angle = torch.cuda.FloatTensor(anglesize).fill_(-90 / 180 * math.pi)
        # Hoek van 0 graden (vanuitgaand dat mensen +- recht zitten/staan)
        angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Vervormingen toepassen aan de hand van een affine grid
        scale_x = scale_x.view(anglesize)
        scale_y = scale_y.view(anglesize)

        tx = (-target_x +0.5)*2
        ty = (-target_y +0.5)*2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos/scale_x
        theta[:, 0, 1] = sin/scale_y
        theta[:, 0, 2] = tx*cos/scale_x+ty*sin/scale_y
        theta[:, 1, 0] = -sin/scale_x
        theta[:, 1, 1] = cos/scale_y
        theta[:, 1, 2] = -tx*sin/scale_x+ty*cos/scale_y

        grid = F.affine_grid(theta, adv_patch_torso.shape)

        adv_batch_torso = F.grid_sample(adv_patch_torso, grid)
        msk_batch_torso = F.grid_sample(msk_batch, grid)

        # De verschillende arm patches bepalen op ongeveer dezelfde manier als het torso
        # Armen hebben maar 2 punten -> dikte geschat
        arm_width = torch.abs(lab_batch[:, :, 16] - lab_batch[:, :, 19])/2.5+1
        arm_l_top = lab_batch.narrow(2, 16, 2)
        arm_l_bottom = lab_batch.narrow(2, 22, 2)

        adv_patch_boven_arm_l, msk_batch_boven_arm_l = self.arm_forward(adv_patch_boven_arm_l, img_size, arm_width, arm_l_top, arm_l_bottom, anglesize, msk_batch, current_patch_size, batch_size)

        arm_r_top = lab_batch.narrow(2, 19, 2)
        arm_r_bottom = lab_batch.narrow(2, 25, 2)

        adv_patch_boven_arm_r, msk_batch_boven_arm_r = self.arm_forward(adv_patch_boven_arm_r, img_size, arm_width, arm_r_top, arm_r_bottom, anglesize, msk_batch, current_patch_size, batch_size)

        arm_l_top = lab_batch.narrow(2, 22, 2)
        arm_l_bottom = lab_batch.narrow(2, 28, 2)

        adv_patch_onder_arm_l, msk_batch_onder_arm_l = self.arm_forward(adv_patch_onder_arm_l, img_size, arm_width, arm_l_top, arm_l_bottom, anglesize, msk_batch, current_patch_size, batch_size)

        arm_r_top = lab_batch.narrow(2, 25, 2)
        arm_r_bottom = lab_batch.narrow(2, 31, 2)

        adv_patch_onder_arm_r, msk_batch_onder_arm_r = self.arm_forward(adv_patch_onder_arm_r, img_size, arm_width, arm_r_top, arm_r_bottom, anglesize, msk_batch, current_patch_size, batch_size)

        # Alle patches en masks samenvoegen
        adv_batch_torso = torch.where((adv_batch_torso == 0), adv_patch_boven_arm_l, adv_batch_torso)
        msk_batch_torso = torch.where((msk_batch_torso == 0), msk_batch_boven_arm_l, msk_batch_torso)

        adv_batch_torso = torch.where((adv_batch_torso == 0), adv_patch_boven_arm_r, adv_batch_torso)
        msk_batch_torso = torch.where((msk_batch_torso == 0), msk_batch_boven_arm_r, msk_batch_torso)

        adv_batch_torso = torch.where((adv_batch_torso == 0), adv_patch_onder_arm_l, adv_batch_torso)
        msk_batch_torso = torch.where((msk_batch_torso == 0), msk_batch_onder_arm_l, msk_batch_torso)

        adv_batch_torso = torch.where((adv_batch_torso == 0), adv_patch_onder_arm_r, adv_batch_torso)
        msk_batch_torso = torch.where((msk_batch_torso == 0), msk_batch_onder_arm_r, msk_batch_torso)

        adv_batch = adv_batch_torso.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch = msk_batch_torso.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.999999)

        return adv_batch * msk_batch

    def arm_forward(self, adv_patch_arm, img_size, width, kp_top, kp_bottom, anglesize, msk_batch, current_patch_size, batch_size):

        target_size_x = width * self.patch_scale*0.9

        target_size_y = torch.sqrt((kp_top[:, :, 0] - kp_bottom[:, :, 0])**2 + (kp_top[:, :, 1] - kp_bottom[:, :, 1])**2)+1
        target_size_y *= self.patch_scale*0.9

        # Wanneer de armen niet volledig geannoteerd zijn krijgt men anders uitschieters
        target_size_y = torch.where(target_size_y > 6*target_size_x, target_size_y/target_size_y, target_size_y)

        # print("SX", target_size_x)
        # print("SY", target_size_y)

        scale_x = target_size_x / current_patch_size
        scale_y = target_size_y / current_patch_size

        target_x = (((kp_top[:, :, 0] + kp_bottom[:, :, 0])/2)/img_size).view(np.prod(batch_size))
        target_y = (((kp_top[:, :, 1] + kp_bottom[:, :, 1])/2)/img_size).view(np.prod(batch_size))

        # print("LB", lab_batch[0:0,:])
        #
        # print("TX", target_x)
        # print("TY", target_y)

        s = adv_patch_arm.size()
        adv_patch_arm = adv_patch_arm.view(s[0] * s[1], s[2], s[3], s[4])

        angle_degrees = (torch.tan((kp_top[:, :, 0] - kp_bottom[:, :, 0])/(kp_top[:, :, 1] - kp_bottom[:, :, 1]))*57.2958)
        angle_90 = torch.cuda.FloatTensor(angle_degrees.size(0) * angle_degrees.size(1)).fill_(90)
        angle_degrees = angle_degrees.view(angle_degrees.size(0) * angle_degrees.size(1))
        angle_degrees = torch.where(torch.isnan(angle_degrees), angle_90, angle_degrees)

        # angle = torch.cuda.FloatTensor(anglesize).fill_(45)
        # print("ANGLE", angle_degrees[0])

        scale_x = scale_x.view(anglesize)
        scale_y = scale_y.view(anglesize)

        tx = (-target_x +0.5)*2
        ty = (-target_y +0.5)*2

        sin = torch.sin(angle_degrees)
        cos = torch.cos(angle_degrees)

        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos/scale_x
        theta[:, 0, 1] = sin/scale_y
        theta[:, 0, 2] = tx*cos/scale_x+ty*sin/scale_y
        theta[:, 1, 0] = -sin/scale_x
        theta[:, 1, 1] = cos/scale_y
        theta[:, 1, 2] = -tx*sin/scale_x+ty*cos/scale_y

        grid = F.affine_grid(theta, adv_patch_arm.shape)

        adv_patch_arm = F.grid_sample(adv_patch_arm, grid)
        msk_batch_arm = F.grid_sample(msk_batch, grid)

        return adv_patch_arm, msk_batch_arm


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7,same=True)
        self.patch_scale = 1.5
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)#.unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()


        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()


        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor


        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        print(cls_ids)

        s = adv_batch.size()
        img = msk_batch.view(s[0], s[1], s[2], s[3], s[4])[0, 0, :, :, :].detach().cpu()
        img = transforms.ToPILImage()(img)
        img.show()

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)


        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        # print("1", target_size)
        target_size *= self.patch_scale
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if(rand_loc):
            off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
            target_x = target_x + off_x
            off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            target_y = target_y + off_y
        target_y = target_y - 0.05
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)
        # print("2", target_size)
        print("1", target_x)
        print("2", target_y)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        img = msk_batch.view(s[0], s[1], s[2], s[3], s[4])[0, 0, :, :, :].detach().cpu()
        img = transforms.ToPILImage()(img)
        img.show()

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2

        print("TX", tx)
        print("TY", ty)

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -ty*sin/scale+cos/scale

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)


        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        img = transforms.ToPILImage()(img)
        img.show()
        exit()

        return adv_batch_t * msk_batch_t



class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch

'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''

class CocoKeypointDataset(Dataset):
    # Een pytorch dataloader voor de COCO keypoints dataset
    def __init__(self, root, json, imgsize, max_lab, transform=None):
        self.root = root
        # Er wordt gebruik gemaakt van de COCO API
        self.coco = COCO(json)
        cats = self.coco.loadCats(self.coco.getCatIds())
        catIds = self.coco.getCatIds(catNms=['person']);
        self.ids = self.coco.getImgIds(catIds=catIds)
        self.transform = transform
        self.imgsize = imgsize
        self.maxlab = max_lab

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'

        coco = self.coco
        # We halen een afbeelding op met index "idx"
        img_info = coco.loadImgs(self.ids[idx])[0]
        # Deze afbeelding wordt geopend
        image = Image.open(os.path.join(self.root, img_info['file_name'])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # Nu halen we de bijhorende annotations op
        annIds = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(annIds)
        # We geven enkel het keypoint gedeelte verder door
        anns = self.pad(anns)
        # We herschalen de afbeelding en annotations naar de img_size van yolo
        image, anns = self.rescale(image, anns)
        transform = transforms.ToTensor()
        image = transform(image)

        anns = torch.Tensor(anns)
        return image, anns

    def __len__(self):
        return len(self.ids)

    def rescale(self, img, anns):
        # Hier zullen we de afbeelding en annotations herschalen naar een vierkant van img_size x img_size
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img)
            else:
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img)

        for i in range (self.maxlab):
            for j in range (1, 52, 3):
                if w > h:
                    anns[i][j] = anns[i][j]*self.imgsize/w
                    anns[i][j+1] = anns[i][j+1]*self.imgsize/w
                else:
                    anns[i][j] = anns[i][j]*self.imgsize/h
                    anns[i][j+1] = anns[i][j+1]*self.imgsize/h

        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)
        return padded_img, anns

    def pad(self, anns_list):
        keypoint_anns = []
        for x in anns_list:
            # print(x['keypoints'])
            if x['keypoints'][17] > 0 and x['keypoints'][20] > 0:
                keypoint_anns.append(list([0] + x['keypoints']))
        # Hier worden de annotations bijgevuld zodat elke afbeelding er evenveel heeft
        pad_size = self.maxlab - len(keypoint_anns)
        if(pad_size>0):
            padded_lab = keypoint_anns
            for i in range(pad_size):
                padded_lab.append([1]+[0] * 51)
        else:
            if(pad_size < 0):
                padded_lab = keypoint_anns[:self.maxlab]
            else:
                padded_lab = keypoint_anns
        return padded_lab


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab


if __name__ == '__main__':
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()

    test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                              batch_size=3, shuffle=True)

    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolov2.weights"
    printfile = "non_printability/30values.txt"

    patch_size = 400

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80).cuda()
    nms_calculator = NMSCalculator(printfile, patch_size)
    total_variation = TotalVariation()
    '''
    img = Image.open('data/horse.jpg').convert('RGB')
    img = img.resize((darknet_model.width, darknet_model.height))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = torch.autograd.Variable(img)

    output = darknet_model(img)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    tl0 = time.time()
    tl1 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        tl1 = time.time()
        print('time to fetch items: ',tl1-tl0)
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch = Image.open('data/horse.jpg').convert('RGB')
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        img_size = img_batch.size(-1)
        print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer.forward(adv_patch, lab_batch, img_size)
        print('applying patches')
        t1 = time.time()
        img_batch = patch_applier.forward(img_batch, adv_batch_t)
        img_batch = torch.autograd.Variable(img_batch)
        img_batch = F.interpolate(img_batch,(darknet_model.height, darknet_model.width))
        print('running patched images through model')
        t2 = time.time()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    try:
                        print(type(obj), obj.size())
                    except:
                        pass
            except:
                pass

        print(torch.cuda.memory_allocated())

        output = darknet_model(img_batch)
        print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        t4 = time.time()
        nms = nms_calculator.forward(adv_patch)
        tv = total_variation(adv_patch)
        print('---------------------------------')
        print('        patch transformation : %f' % (t1-t0))
        print('           patch application : %f' % (t2-t1))
        print('             darknet forward : %f' % (t3-t2))
        print('      probability extraction : %f' % (t4-t3))
        print('---------------------------------')
        print('          total forward pass : %f' % (t4-t0))
        del img_batch, lab_batch, adv_patch, adv_batch_t, output, max_prob
        torch.cuda.empty_cache()
        tl0 = time.time()
